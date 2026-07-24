from io import BytesIO

from PIL import Image

from fastapi import APIRouter
from fastapi import File
from fastapi import Form
from fastapi import UploadFile

from app.dependencies import predictor

router = APIRouter()


@router.post("/predict")
async def predict(
    text: str = Form(...),
    image: UploadFile = File(...),
):

    image_bytes = await image.read()

    image = Image.open(
        BytesIO(image_bytes)
    )

    result = predictor.predict(
        text,
        image,
    )

    return result