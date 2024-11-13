import flet as ft
import websockets
import asyncio
import base64
import json
from PIL import Image 
import io
import os

def main(page: ft.Page):
    page.title = "Rice Plant Disease Detection"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 30
    page.window.width = 1000
    page.window.height = 800
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    
    # Header section
    header = ft.Container(
        content=ft.Column(
            [
                ft.Text(
                    "Rice Plant Disease Detection",
                    size=32,
                    weight=ft.FontWeight.BOLD,
                    color="#2E7D32"
                ),
                ft.Text(
                    "Upload an image to analyze plant health",
                    size=16,
                    color="#666666"
                )
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=5
        ),
        margin=ft.margin.only(bottom=30)
    )

    # Create image holder with placeholder
    image_holder = ft.Image(
        visible=False,
        fit=ft.ImageFit.CONTAIN,
        border_radius=ft.border_radius.all(8)
    )
    
    result_mobilenet = ft.Text("", size=16, text_align=ft.TextAlign.CENTER)
    result_resnet = ft.Text("", size=16, text_align=ft.TextAlign.CENTER)
    
    # Function to handle file selection
    def handle_loaded_file(e: ft.FilePickerResultEvent):
        if e.files and len(e.files):
            file_path = e.files[0].path
            image_holder.src = file_path
            image_holder.visible = True
            
            # Reset results when new image is selected
            result_mobilenet.value = ""
            result_resnet.value = ""
            
            # Enable analyze button
            analyze_button.disabled = False
            page.update()

    # File picker setup with initial directory
    initial_directory = os.path.expanduser(r"C:\Users\Noelle\Downloads\CPE178P_Group02_Project\client\Rice_Leaf_for_Testing")
    filepick = ft.FilePicker(
        on_result=handle_loaded_file,
    )
    page.overlay.append(filepick)

    # Function to handle prediction
    def analyze_image(e):
        if image_holder.src:
            with open(image_holder.src, "rb") as image_file:
                image_bytes = image_file.read()
                image_data = base64.b64encode(image_bytes).decode("utf-8")
            
            # Show loading state
            result_mobilenet.value = "Analyzing..."
            result_resnet.value = "Analyzing..."
            page.update()
            
            asyncio.run(send_prediction_request(image_data))

    async def send_prediction_request(image_data):
        try:
            async with websockets.connect("ws://localhost:8000/ws") as websocket:
                await websocket.send(json.dumps({
                    "type": "predict",
                    "data": image_data
                }))

                response = await websocket.recv()
                data = json.loads(response)

                if data.get("type") == "prediction":
                    # Format results with confidence scores
                    mobilenet_result = (
                        f"MobileNetV2 Prediction:\n"
                        f"{data.get('class')}\n"
                        f"Score: {data.get('output')}%"
                    )
                    
                    resnet_result = (
                        f"ResNet50 Prediction:\n"
                        f"{data.get('class')}\n"
                        f"Score: {data.get('output_rn')}%"
                    )
                    
                    result_mobilenet.value = mobilenet_result
                    result_resnet.value = resnet_result
                else:
                    result_mobilenet.value = "Error occurred during analysis"
                    result_resnet.value = "Error occurred during analysis"
        except Exception as e:
            result_mobilenet.value = f"Connection error: {str(e)}"
            result_resnet.value = f"Connection error: {str(e)}"
        page.update()

    # Create main content layout
    image_section = ft.Container(
        content=ft.Column(
            [
                ft.Container(
                    content=image_holder,
                    bgcolor="#F5F5F5",
                    padding=20,
                    border_radius=10,
                    width=400,
                    height=400,
                    alignment=ft.alignment.center,
                ),
                ft.FilledTonalButton(
                    "Choose a Photo",
                    icon=ft.icons.UPLOAD_FILE,
                    on_click=lambda _: filepick.pick_files(
                        allow_multiple=False,
                        allowed_extensions=['jpg', 'png', 'jpeg']
                    ),
                    width=200
                )
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=20
        ),
        margin=ft.margin.only(right=30)
    )

    # Results section
    results_section = ft.Container(
        content=ft.Column(
            [
                ft.Container(
                    content=result_mobilenet,
                    bgcolor="#E8F5E9",
                    padding=20,
                    border_radius=10,
                    width=300,
                    height=150,
                    alignment=ft.alignment.center,
                ),
                ft.Container(
                    content=result_resnet,
                    bgcolor="#E8F5E9",
                    padding=20,
                    border_radius=10,
                    width=300,
                    height=150,
                    alignment=ft.alignment.center,
                )
            ],
            spacing=20
        )
    )

    analyze_button = ft.ElevatedButton(
        "Analyze Rice Plant Image",
        icon=ft.icons.ANALYTICS,
        on_click=analyze_image,
        disabled=True,
        style=ft.ButtonStyle(
            color="white",
            bgcolor="#2E7D32",
            padding=20,
        ),
        width=200
    )

    content = ft.Column(
        [
            header,
            ft.Row(
                [image_section, results_section],
                alignment=ft.MainAxisAlignment.CENTER
            ),
            ft.Container(
                content=analyze_button,
                alignment=ft.alignment.center,
                margin=ft.margin.only(top=20)
            )
        ],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER
    )

    page.add(content)

ft.app(target=main)
