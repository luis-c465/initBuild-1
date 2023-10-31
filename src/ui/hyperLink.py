import webbrowser

from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.uix.label import Label

"""A kivy widget that implements a hyperlink"""

KV = """

<MyHyperlink>:

"""

Builder.load_string(KV)


class MyHyperlink(Label):
    target = StringProperty("")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        kwargs["markup"] = True
        kwargs["color"] = (0, 0, 1, 1)
        kwargs["text"] = "[u][ref=link]{}[/ref][/u]".format(kwargs["text"])
        kwargs["on_ref_press"] = self.link

    def link(self, *args):
        webbrowser.open(self.target)
