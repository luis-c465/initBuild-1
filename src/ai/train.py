import time
from datetime import timedelta
from threading import Thread
from typing import Callable

from humanize import naturaldelta
from IPython.core.magics.execution import _format_time

from src.ai.trainer import Trainer


class ThreadedTrainer(Thread):
    trainer: Trainer

    def __init__(
        self,
        env_name: str,
        on_epoch: Callable[[float, float], None],
        on_update: Callable[[str], None],
        on_done: Callable[[], None],
        *args,
        **kwargs,
    ):
        super().__init__()

        self.on_epoch = on_epoch
        self.on_done = on_done
        self.on_update = on_update

        self.trainer = Trainer(env_name, logging=True, *args, **kwargs)

    def run(self):
        self.on_update("Warming up")
        self.trainer.warm_up()
        self.on_update("Warm up done... Starting training")

        start_time = time.monotonic()
        for n_epoch in range(self.trainer.epochs):
            total_reward, total_loss = self.trainer.epoch(n_epoch)
            end_time = time.monotonic()
            delta = timedelta(seconds=end_time - start_time)

            self.on_update(
                f"Epoch #{n_epoch} -> R:[b]{total_reward:.2f}[/b] L:[b]{total_loss:.2f}[/b] T:{_format_time(delta.microseconds / 100_000)}"
            )

            self.on_epoch(total_reward, total_loss)
            start_time = time.monotonic()

        self.on_done()
