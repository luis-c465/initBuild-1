from kivy.lang import Builder
from kivymd.app import MDApp


class MainApp(MDApp):
    def build(self):
        return Builder.load_file("src/ui/app.kv")

    def on_start(self):
        self.fps_monitor_start()


MainApp().run()
