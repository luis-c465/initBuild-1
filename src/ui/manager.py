import multiprocessing
import time
from queue import Empty

from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import NumericProperty
from kivymd.app import MDApp
from kivymd.uix.button import MDFlatButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.screenmanager import ScreenManager

from ai import train
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
    training: multiprocessing.Process

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
                    on_press=lambda env_id=env_id, *_, **__: self.begin_training(
                        env_id
                    ),
                ),
            ],
        )
        self.dialog.open()

    def begin_training(self, env_id: str, *_, **__):
        self.dialog.dismiss()
        self.current = "progress"

        self.queue = multiprocessing.Queue()
        self.event = multiprocessing.Event()
        self.training = multiprocessing.Process(
            target=train,
            args=(
                self.queue,
                self.event,
                str(env_id),
                "mps",
            ),
        )
        self.training.start()

        while not self.event.is_set():
            try:
                progress = self.queue.get(
                    timeout=0.050
                )  # Wait for 1 second for progress update
                print(
                    progress
                )  # Update your loading screen with this progress information
            except Empty:
                pass

        self.training.join()  # Wait for the training process to finish before exiting

    def on_start(self, *_, **__):
        grid = self.ids["images_grid"]

        for game in gameList.values():
            g = Game(game, on_press=self.on_game_press)
            g.on_press = self.on_game_press
            grid.add_widget(g)
