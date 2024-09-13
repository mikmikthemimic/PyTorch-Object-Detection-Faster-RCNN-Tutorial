import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.create_viewport(title='Sample Output', width=600, height=300)

with dpg.window(label="Sample Application", width=300, height=150):
    dpg.add_text("Test")

dpg.setup_dearpygui()
dpg.show_viewport()

dpg.start_dearpygui()

#while loop below replaces start_dearpygui
#while dpg.is_dearpygui_running():

dpg.destroy_context()