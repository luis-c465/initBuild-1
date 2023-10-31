from kivy.clock import Clock
from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.button import MDFlatButton
from kivymd.uix.dialog import MDDialog

from src.gameList import GameDict, gameList
from src.ui.game import Game
from ui.gameModal import GameModal

# from src.ui.game import Game

KV = """
MDScreen:
    MDLabel:
        text: "Hello, World!"
        halign: "center"

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

"""


class MainApp(MDApp):
    dialog: MDDialog = None

    def on_game_press(self, data: GameDict, *_, **__):
        env_id = data["env"]

        if self.dialog:
            self.dialog.dismiss()
            self.dialog = None

        self.dialog = MDDialog(
            title=data["name"],
            height="500dp",
            type="custom",
            content_cls=GameModal(data),
            buttons=[
                MDFlatButton(
                    text="Cancel",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.secondary_text_color,
                    on_press=lambda *_, **__: self.dialog.dismiss(),
                ),
                MDFlatButton(
                    text="Start Training",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.primary_color,
                    on_press=lambda *_, **__: print(env_id),
                ),
            ],
        )
        self.dialog.open()

    def build(self):
        Clock.schedule_once(self.on_start, 5)

        return Builder.load_string(KV)

    def on_start(self, *_, **__):
        grid = self.root.ids["images_grid"]

        for game in gameList.values():
            g = Game(game, on_press=self.on_game_press)
            g.on_press = self.on_game_press
            grid.add_widget(g)


MainApp().run()
