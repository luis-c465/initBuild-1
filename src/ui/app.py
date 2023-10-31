from kivy.lang import Builder
from kivymd.app import MDApp

import ui.manager

# from src.ui.game import Game

KV = """
Manager:

"""


class MainApp(MDApp):
    def build(self):
        return Builder.load_string(KV)


MainApp().run()
