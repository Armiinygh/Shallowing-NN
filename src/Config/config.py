from types import SimpleNamespace

cfg = SimpleNamespace(**{})

cfg.epochs = 20
cfg.batch_size = 128
cfg.learning_rate = 0.00001
cfg.output_size = 256

# run config
cfg.save_model = True