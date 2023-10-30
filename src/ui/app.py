from kivy.clock import Clock
from kivy.lang import Builder
from kivymd.app import MDApp

from src.gameList import gameList

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
            height: self.minimum_height
            width: self.minimum_width
            spacing: 10
            row_default_height: "500dp"
            row_force_default: True

            Image:
                source: "src/assets/zaxxon.zip"
                size_hint: None, None
                size: 500, 500
                allow_stretch: True
                keep_ratio: False
                anim_delay: 0
                anim_loop: -1

"""


class MainApp(MDApp):
    def build(self):
        Clock.schedule_once(self.on_start, 5)

        return Builder.load_string(KV)

    def on_start(self, *_, **__):
        grid = self.root.ids["images_grid"]

        # for game in gameList.values():
        #     grid.add_widget(Game(game))


MainApp().run()
