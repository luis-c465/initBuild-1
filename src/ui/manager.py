import multiprocessing
import time
from queue import Empty
from threading import Thread

from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import NumericProperty
from kivymd.app import MDApp
from kivymd.uix.button import MDFlatButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.screenmanager import ScreenManager

from src.ai.train import ThreadedTrainer
from src.gameList import GameDict, gameList
from src.ui.game import Game
from ui.gameModal import GameModal

KV = """
<Manager>:
    id: manager

    MDScreen:
        name: "main"
        MDScrollView:
            MDGridLayout:
                cols: 5
                id: images_grid
                size_hint_y: None
                height: self.minimum_height  #<<<<<<<<<<<<<<<<<<<<
                spacing: 10
                row_default_height: "300dp"
                col_default_width: "200dp"
                col_force_default: True
    MDScreen:
        name: "progress"

        MDProgressBar:
            orientation: "vertical"
            value: root.progress
"""

Builder.load_string(KV)


class Manager(ScreenManager):
    dialog: MDDialog = None
    progress = NumericProperty(0)

    queue: multiprocessing.Queue

    event: "multiprocessing.Event"
    trainer: ThreadedTrainer
    training: bool = False

    def __init__(self, **kwargs):
        Clock.schedule_once(self.on_start)

        super().__init__(**kwargs)

    def on_game_press(self, data: GameDict, *_, **__):
        env_id = data["env"]

        if self.dialog:
            self.dialog.dismiss()
            self.dialog = None

        app = MDApp.get_running_app()
        self.dialog = MDDialog(
            title=data["name"],
            height="500dp",
            type="custom",
            content_cls=GameModal(data),
            buttons=[
                MDFlatButton(
                    text="Cancel",
                    theme_text_color="Custom",
                    text_color=app.theme_cls.secondary_text_color,
                    on_press=lambda *_, **__: self.dialog.dismiss(),
                ),
                MDFlatButton(
                    text="Start Training",
                    theme_text_color="Custom",
                    text_color=app.theme_cls.primary_color,
                    on_press=lambda *_, **__: self.begin_training(env_id),
                ),
            ],
        )
        self.dialog.open()

    def begin_training(self, env_id: str, *_, **__):
        self.dialog.dismiss()
        self.current = "progress"

        if self.training:
            return

        self.trainer = ThreadedTrainer(
            env_id, on_epoch=self.on_epoch, on_done=self.on_done
        )
        self.trainer.start()
        self.training = True

    def on_epoch(self, total_reward: float, total_loss: float):
        print(total_reward, total_loss)

    def on_done(self):
        print("done training")
        self.training = False

    def on_start(self, *_, **__):
        grid = self.ids["images_grid"]

        for game in gameList.values():
            g = Game(game, on_press=self.on_game_press)
            g.on_press = self.on_game_press
            grid.add_widget(g)
