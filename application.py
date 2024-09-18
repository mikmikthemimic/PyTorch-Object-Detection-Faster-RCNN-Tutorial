import dearpygui.dearpygui as dpg

from model import get_input

dpg.create_context()

def get_predictions(json_file_name, image_name):
    # get predictions json file from predict()
    # get image with predictions
    add_and_load_image(final_image, parent="Primary Window")


# Callbacks
def select_image(sender, app_data, user_data):
    selections = app_data.get("selections")
    image = next(iter(selections.values()))
    print(f"App Data: {app_data}")
    print(f"Image: {image}")

    image_name = next(iter(selections.keys()))

    if dpg.does_item_exist("image_tag"):
        print("Deleting existing image")
        dpg.delete_item("image_tag")
        dpg.delete_item("texture_tag")
    
    #TODO: FIGURE OUT KUNG ANO ICACALL HERE BC IT CANNOT BE GET_INPUT, GET_INPUT WILL BE CALLED BY ANOTHER METHOD SA MODEL.PY; IPASS NALANG DAW YUNG IMAGE TO GET_INPUT
    predict(image_name)

# Lambdas
def add_and_load_image(image_path, parent=None):
    width, height, channels, data = dpg.load_image(image_path)

    with dpg.texture_registry() as reg_id:
        texture_id = dpg.add_static_texture(width, height, data, parent=reg_id, tag="texture_tag")

    if parent is None:
        return dpg.add_image(texture_id, tag="image_tag")
    else:
        return dpg.add_image(texture_id, parent=parent, tag="image_tag")

with dpg.file_dialog(
    tag="file_dialog_id",
    directory_selector=False,
    show=False,
    file_count=1,
    height=300,
    callback=select_image
):
    dpg.add_file_extension(".*")
    dpg.add_file_extension(".jpg", color=(0, 255, 0, 255))
    dpg.add_file_extension(".png", color=(0, 255, 0, 255))
    dpg.add_file_extension(".jpeg", color=(0, 255, 0, 255))

# Main Window
with dpg.window(
    label="Main Window",
    tag="Primary Window",
    width=1280,
    height=840,
    no_resize=True,
    no_collapse=True,
    no_move=True,
    no_scroll_with_mouse=False,
    no_scrollbar=False,
    no_title_bar=True,
    no_background=False,
):
    dpg.add_button(label="Select Image", callback=lambda: dpg.show_item("file_dialog_id"))

# Setup
dpg.create_viewport(
    title="Test", width=1280, height=840, x_pos=0, y_pos=0, resizable=False
)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()