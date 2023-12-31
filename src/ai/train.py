import threading
import time
from datetime import timedelta
from threading import Thread
from typing import Callable

from humanize import naturaldelta
from IPython.core.magics.execution import _format_time

from src.ai.trainer import Trainer


class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition.

    From stack overflow -> https://stackoverflow.com/a/325528
    """

    def __init__(self, *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class ThreadedTrainer(StoppableThread):
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
            if self.stopped():
                self.on_update("Stopping training")
                self.trainer.save_and_close()
                break

            total_reward, total_loss = self.trainer.epoch()
            end_time = time.monotonic()
            delta = timedelta(seconds=end_time - start_time)

            self.on_update(
                f"Epoch #{n_epoch} -> R:[b]{total_reward:.2f}[/b] L:[b]{total_loss:.2f}[/b] T:{_format_time(delta.microseconds / 100_000)}"
            )

            self.on_epoch(total_reward, total_loss)
            start_time = time.monotonic()

        self.trainer.save_and_close()
        self.on_done()
