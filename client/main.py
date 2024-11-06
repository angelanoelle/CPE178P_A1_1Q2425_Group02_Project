import flet as ft
import websockets
import asyncio
import base64
import json
from PIL import Image 
import io
import re

def main(page: ft.Page):
    page.title = "Flower Classification"
    page.scroll = "adaptive"
    # page.window_width = 800   # deprecated
    # page.window_height = 500  # deprecated

    page.window.width = 800
    page.window.height = 500
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    # page.window_resizable = False

    textMS = ft.Text("MindSpore Flower Image Detection", color="grey", theme_style=ft.TextThemeStyle.DISPLAY_SMALL)
    # image_holder = ft.Image(visible=False, fit=ft.ImageFit.CONTAIN)
    image_holder = ft.Image(visible=False)
    
    result_text = ft.Text("")
    result_resnet = ft.Text("")
    import tempfile

    def handle_loaded_file(e: ft.FilePickerResultEvent):
        if e.files and len(e.files):
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                # Read the image file in binary mode
                with open(e.files[0].path, "rb") as image_file:
                    temp_file.write(image_file.read())
                    # Set the src_base64 of the image holder
                image_holder.src = temp_file.name
                image_holder.visible = True
                page.update()

            # Get the file path (you already have it from e.files[0].path)
            file_path = e.files[0].path  
            print(file_path)

            # Load the image using Pillow
            img = Image.open(file_path)
            # Save the image to bytes
            byte_io = io.BytesIO()
            img.save(byte_io, 'PNG')  # or 'JPEG' depending on the image format
            byte_io.seek(0)  # Reset the stream position

            # Now you can use byte_io.read() to get the bytes
            image_bytes = byte_io.read() 

            image_data = base64.b64encode(image_bytes).decode("utf-8")
                # asyncio.run(send_prediction_request(image_data))

    filepick=ft.FilePicker(on_result=handle_loaded_file)
    page.overlay.append(filepick)

    def predict_image(e):
        if image_holder.src:
            # Read the image data from the file pointed to by image_holder.src
            with open(image_holder.src, "rb") as image_file:
                image_bytes = image_file.read()
                image_data = base64.b64encode(image_bytes).decode("utf-8")

            asyncio.run(send_prediction_request(image_data))
        else:
            print("No image selected")                        

    async def send_prediction_request(image_data):
        async with websockets.connect("ws://localhost:8000/ws") as websocket:
            await websocket.send(json.dumps({
                "type": "predict",
                "data": image_data
            }))

            response = await websocket.recv()
            data = json.loads(response)

            if data.get("type") == "prediction":
                result_text.value = f"Predicted Class: {data.get('class')} ({data.get('output')})"
                result_resnet.value = f"Predicted Class: {data.get('class_rn')} ({data.get('output_rn')})"
                selected_image.controls[2].controls[0].content.value = f"Predicted Class using MobileNetV2: \n{data.get('class')} \nScore: {data.get('output')}"
                selected_image.controls[2].controls[1].content.value = f"Predicted Class using Resnet50: \n{data.get('class_rn')} \nScore: {data.get('output_rn')}"
            else:
                result_text.value = "Error occurred during prediction"
                result_resnet.value = "Error occurred during prediction"
                selected_image.controls[2].controls[0].content.value = "Error occurred during prediction"
                selected_image.controls[2].controls[1].content.value = "Error occurred during prediction"
        page.update()

    selected_image=ft.Row(
        [
            ft.Container(
                content=image_holder,
                margin=10,
                padding=10,
                border=ft.border.all(5, ft.colors.BLACK),
                alignment=ft.alignment.center,
                bgcolor=ft.colors.WHITE,
                width=250,
                height=250,
                border_radius=10,
                ink=True,
                on_click=lambda _:filepick.pick_files (\
                    allow_multiple=False, allowed_extensions=['jpg', 'png', 'jpeg']),
            ),
            ft.Container(
                content=ft.Image(
                    src=f"arrowtotheright.png",
                    height=160,
                    fit=ft.ImageFit.CONTAIN,
                )
            ),
            ft.Column(
            [
                ft.Container(
                    content=result_text,
                    margin=10,
                    padding=10,
                    border=ft.border.all(5, ft.colors.BLACK),
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.GREY,
                    width=300,
                    height=125,
                    border_radius=10,
                ),
                ft.Container(
                    content=result_resnet,
                    margin=10,
                    padding=10,
                    border=ft.border.all(5, ft.colors.BLACK),
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.GREY,
                    width=300,
                    height=125,
                    border_radius=10,
                ),
            ]
            )
        ],
        alignment=ft.MainAxisAlignment.CENTER
    )

    predict_button = ft.Container(
        ft.ElevatedButton(text="Predict", width=150, height=50, on_click= predict_image),
        alignment=ft.alignment.center,
    )
    
    
   
    page.add(
        selected_image,
        predict_button
    )

ft.app(target=main)