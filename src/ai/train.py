from threading import Thread
from typing import Callable

from src.ai.trainer import Trainer


class ThreadedTrainer(Thread):
    trainer: Trainer

    def __init__(
        self,
        env_name: str,
        on_epoch: Callable[[float, float], None],
        on_done: Callable[[], None],
        *args,
        **kwargs
    ):
        super().__init__()

        self.on_epoch = on_epoch
        self.on_done = on_done

        self.trainer = Trainer(env_name, logging=True, *args, **kwargs)

    def run(self):
        self.trainer.warm_up()
        print("Warm up done")

        for n_epoch in range(self.trainer.epochs):
            total_reward, total_loss = self.trainer.epoch(n_epoch)

            self.on_epoch(total_reward, total_loss)

        self.on_done()
