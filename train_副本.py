from google.colab import drive
drive.mount('/content/drive')

import os
import logging
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

policy = tf.keras.mixed_precision.Policy('float32')
tf.keras.mixed_precision.set_global_policy(policy)
logger.info(f"精度策略已设置为: {policy.name}")

class Config:
    DRIVE_BASE_PATH = '/content/drive/My Drive'
    PROJECT_NAME = 'YoochooseProject'
    PROCESSED_DATA_PATH = os.path.join(DRIVE_BASE_PATH, PROJECT_NAME, 'processed_tfrecord')
    MODEL_PATH = os.path.join(DRIVE_BASE_PATH, PROJECT_NAME, 'models_yoochoose')

    BATCH_SIZE = 256
    GRADIENT_ACCUMULATION_STEPS = 1
    EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 7
    PREFETCH_BUFFER = tf.data.AUTOTUNE
    SHUFFLE_BUFFER = 50000

    INITIAL_LR = 1e-4
    WARMUP_EPOCHS = 5

    DROPOUT_RATE = 0.2
    LABEL_SMOOTHING = 0.05

os.makedirs(Config.MODEL_PATH, exist_ok=True)

class NanTerminator(Callback):
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                print(f"\n发现NaN/Inf loss在batch {batch}，停止训练")
                self.model.stop_training = True

def create_yoochoose_transformer_model(
    vocab_size_items: int,
    vocab_size_cats: int,
    max_len: int,
    d_model: int = 128,
    num_heads: int = 4,
    num_blocks: int = 2,
    ff_dim_multiplier: int = 2,
    dropout_rate: float = 0.2,
    l2_reg: float = 1e-6
):
    items_in = Input(shape=(max_len,), name='items_input')
    cats_in = Input(shape=(max_len,), name='categories_input')
    time_deltas_in = Input(shape=(max_len,), name='time_deltas_input')

    item_embs = Embedding(
        vocab_size_items, d_model,
        mask_zero=True,
        embeddings_initializer='glorot_uniform',
        embeddings_regularizer=tf.keras.regularizers.l2(l2_reg)
    )(items_in)

    cat_embs = Embedding(
        vocab_size_cats, d_model//4,
        mask_zero=True,
        embeddings_initializer='glorot_uniform',
        embeddings_regularizer=tf.keras.regularizers.l2(l2_reg)
    )(cats_in)

    time_deltas_clipped = tf.keras.layers.Lambda(
        lambda x: tf.clip_by_value(x, -10.0, 10.0)
    )(time_deltas_in)
    time_exp = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1))(time_deltas_clipped)
    time_feat = Dense(
        d_model//4,
        kernel_initializer='glorot_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
    )(time_exp)

    x = Concatenate()([item_embs, cat_embs, time_feat])
    x = Dense(
        d_model,
        kernel_initializer='glorot_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
    )(x)

    positions = tf.keras.layers.Lambda(
        lambda inp: tf.tile(
            tf.expand_dims(tf.range(0, max_len, dtype=tf.int32), 0),
            [tf.shape(inp)[0], 1]
        )
    )(items_in)
    pos_embs = Embedding(max_len, d_model, embeddings_initializer='glorot_uniform')(positions)

    x = tf.keras.layers.Add()([x, pos_embs])
    x = Dropout(dropout_rate)(x)

    for block_idx in range(num_blocks):
        residual = x

        attn_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model//num_heads,
            dropout=dropout_rate,
            kernel_initializer='glorot_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )(x, x)
        attn_output = Dropout(dropout_rate)(attn_output)
        x = tf.keras.layers.Add()([residual, attn_output])
        x = LayerNormalization(epsilon=1e-6)(x)

        residual = x
        ffn = Dense(
            d_model * ff_dim_multiplier,
            activation='relu',
            kernel_initializer='glorot_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )(x)
        ffn = Dropout(dropout_rate)(ffn)
        ffn = Dense(
            d_model,
            kernel_initializer='glorot_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )(ffn)
        ffn = Dropout(dropout_rate)(ffn)
        x = tf.keras.layers.Add()([residual, ffn])
        x = LayerNormalization(epsilon=1e-6)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(
        d_model,
        activation='relu',
        kernel_initializer='glorot_uniform',
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
    )(x)
    x = Dropout(dropout_rate)(x)

    output = Dense(1, activation='sigmoid')(x)

    return Model(
        inputs={
            'items_input': items_in,
            'categories_input': cats_in,
            'time_deltas_input': time_deltas_in
        },
        outputs=output
    )

class WarmupExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, warmup_steps, decay_steps, decay_rate):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)

        warmup_lr = self.initial_learning_rate * (step / warmup_steps)

        decay_lr = self.initial_learning_rate * tf.pow(
            self.decay_rate,
            (step - warmup_steps) / self.decay_steps
        )

        return tf.cond(step < warmup_steps, lambda: warmup_lr, lambda: decay_lr)

def create_dataset(
    filepath: str,
    max_sequence_length: int,
    batch_size: int,
    shuffle_buffer: int,
    prefetch_buffer: int,
    training: bool,
    cache_path_prefix: str,
    use_cache: bool = True
):
    feature_description = {
        'items': tf.io.FixedLenFeature([max_sequence_length], tf.int64),
        'categories': tf.io.FixedLenFeature([max_sequence_length], tf.int64),
        'time_deltas': tf.io.FixedLenFeature([max_sequence_length], tf.float32),
        'label': tf.io.FixedLenFeature([1], tf.int64),
    }

    @tf.function
    def _parse_function(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        features = {
            'items_input': parsed['items'],
            'categories_input': parsed['categories'],
            'time_deltas_input': parsed['time_deltas'],
        }
        label = tf.cast(parsed['label'][0], tf.float32)
        return features, label

    dataset = tf.data.TFRecordDataset(filepath, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        dataset = dataset.repeat()

    if use_cache and not training:
        cache_dir = os.path.join(cache_path_prefix, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"cache_{os.path.basename(filepath)}")
        dataset = dataset.cache(cache_file)

    dataset = dataset.batch(batch_size, drop_remainder=training)
    dataset = dataset.prefetch(prefetch_buffer)

    return dataset

def calculate_balanced_class_weights(train_path: str, max_samples: int = 100000) -> dict:
    logger.info("计算类别权重...")

    feature_description = {
        'label': tf.io.FixedLenFeature([1], tf.int64),
    }

    def _parse_label(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        return tf.cast(parsed['label'][0], tf.float32)

    dataset = tf.data.TFRecordDataset(train_path)
    dataset = dataset.map(_parse_label, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.take(max_samples)

    pos_count = 0
    neg_count = 0

    for label in tqdm(dataset, desc="采样计算权重", total=max_samples):
        if label == 1.0:
            pos_count += 1
        else:
            neg_count += 1

    total = pos_count + neg_count
    if total == 0 or pos_count == 0 or neg_count == 0:
        return {0: 1.0, 1: 1.0}

    neg_weight = np.sqrt(total / (2.0 * neg_count))
    pos_weight = np.sqrt(total / (2.0 * pos_count))

    max_weight = 10.0
    neg_weight = min(neg_weight, max_weight)
    pos_weight = min(pos_weight, max_weight)

    return {0: neg_weight, 1: pos_weight}

def main():
    config = Config()

    logger.info("步骤 1: 加载元数据...")
    with open(os.path.join(config.PROCESSED_DATA_PATH, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    logger.info(f"元数据加载成功: {metadata}")

    train_path = os.path.join(config.PROCESSED_DATA_PATH, 'train.tfrecord')
    val_path = os.path.join(config.PROCESSED_DATA_PATH, 'val.tfrecord')
    test_path = os.path.join(config.PROCESSED_DATA_PATH, 'test.tfrecord')

    logger.info("步骤 2: 创建数据集...")
    train_dataset = create_dataset(
        train_path, metadata['max_sequence_length'],
        config.BATCH_SIZE, config.SHUFFLE_BUFFER,
        config.PREFETCH_BUFFER, True, config.PROCESSED_DATA_PATH,
        use_cache=False
    )

    val_dataset = create_dataset(
        val_path, metadata['max_sequence_length'],
        config.BATCH_SIZE, config.SHUFFLE_BUFFER,
        config.PREFETCH_BUFFER, False, config.PROCESSED_DATA_PATH,
        use_cache=True
    )

    test_dataset = create_dataset(
        test_path, metadata['max_sequence_length'],
        config.BATCH_SIZE, config.SHUFFLE_BUFFER,
        config.PREFETCH_BUFFER, False, config.PROCESSED_DATA_PATH,
        use_cache=True
    )

    logger.info("步骤 3: 构建模型...")
    model = create_yoochoose_transformer_model(
        vocab_size_items=metadata['vocab_size_items'],
        vocab_size_cats=metadata['vocab_size_cats'],
        max_len=metadata['max_sequence_length'],
        dropout_rate=config.DROPOUT_RATE,
        l2_reg=1e-6
    )
    model.summary()

    logger.info("步骤 4: 准备训练...")
    try:
        num_train_examples = metadata['num_train_examples']
        logger.info(f"训练样本数: {num_train_examples:,}")
    except KeyError:
        num_train_examples = 20256984
        logger.info(f"使用默认训练样本数: {num_train_examples:,}")

    steps_per_epoch = num_train_examples // config.BATCH_SIZE

    warmup_steps = steps_per_epoch * config.WARMUP_EPOCHS
    lr_schedule = WarmupExponentialDecay(
        initial_learning_rate=config.INITIAL_LR,
        warmup_steps=warmup_steps,
        decay_steps=steps_per_epoch * 5,
        decay_rate=0.9
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(
            from_logits=False,
            label_smoothing=config.LABEL_SMOOTHING
        ),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc_pr', curve='PR'),
            tf.keras.metrics.AUC(name='auc_roc', curve='ROC')
        ]
    )

    class_weight_dict = calculate_balanced_class_weights(train_path)
    logger.info(f"类别权重: {class_weight_dict}")

    callbacks = [
        NanTerminator(),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(config.MODEL_PATH, 'best_model.weights.h5'),
            monitor='val_auc_pr',
            mode='max',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc_pr',
            patience=config.EARLY_STOPPING_PATIENCE,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(config.MODEL_PATH, 'training.log')
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(config.MODEL_PATH, 'logs'),
            histogram_freq=0,
            write_graph=False,
            update_freq='epoch'
        )
    ]

    logger.info(f"步骤 5: 开始训练...")
    logger.info(f"配置: Epochs={config.EPOCHS}, Batch Size={config.BATCH_SIZE}, LR={config.INITIAL_LR}")
    logger.info(f"Steps per Epoch: {steps_per_epoch}")

    try:
        logger.info("测试第一个batch...")
        test_batch = next(iter(train_dataset))
        test_pred = model(test_batch[0], training=False)
        logger.info(f"测试预测: min={tf.reduce_min(test_pred):.4f}, max={tf.reduce_max(test_pred):.4f}")

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=config.EPOCHS,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )

        with open(os.path.join(config.MODEL_PATH, 'history.pkl'), 'wb') as f:
            pickle.dump(history.history, f)

    except Exception as e:
        logger.error(f"训练出错: {e}")
        raise

    logger.info("最终测试集评估...")
    test_metrics = model.evaluate(test_dataset, verbose=1)
    logger.info(f"测试集性能: {dict(zip(model.metrics_names, test_metrics))}")

    tf.keras.backend.clear_session()
    gc.collect()

    logger.info("训练完成！")

if __name__ == "__main__":
    main()