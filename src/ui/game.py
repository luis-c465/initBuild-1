from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.uix.behaviors import ButtonBehavior
from kivymd import uix_path
from kivymd.theming import ThemableBehavior
from kivymd.uix.behaviors import RectangularRippleBehavior
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.fitimage import FitImage
from kivymd.uix.imagelist import MDSmartTile
from kivymd.uix.label import MDLabel
from kivymd.uix.relativelayout import MDRelativeLayout

KV = """
<Game>

    SmartTileImage:
        id: image
        mipmap: root.mipmap
        source: root.source
        radius: root.radius if root.radius else [0, 0, 0, 0]
        size_hint_y: 1 if root.overlap else None
        height: root.height if root.overlap else root.height - box.height
        pos:
            ((0, 0) if root.overlap else (0, box.height)) \
            if root.box_position == "footer" else \
            (0, 0)
        on_release: root.dispatch("on_release")
        on_press: root.dispatch("on_press")

    SmartTileOverlayBox:
        id: box
        md_bg_color: root.box_color
        size_hint_y: None
        padding: "8dp"
        radius: root.box_radius
        height: "68dp" if root.lines == 2 else "48dp"
        pos:
            (0, 0) \
            if root.box_position == "footer" else \
            (0, root.height - self.height)

"""
Builder.load_string(KV)


class AtariImage(RectangularRippleBehavior, ButtonBehavior, FitImage):
    """Implements the tile image."""


class AtariOverlayBox(MDBoxLayout):
    """Implements a container for custom widgets to be added to the tile."""


class Game(MDSmartTile):
    game_id = StringProperty("")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
