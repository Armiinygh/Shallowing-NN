from types import SimpleNamespace

cfg = SimpleNamespace(**{})

cfg.epochs = 20
cfg.batch_size = 10
cfg.learning_rate = 0.1

# run config
cfg.save_model = True