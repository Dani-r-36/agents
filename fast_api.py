from fastapi import FastAPI, Depends, HTTPException, Header, BackgroundTasks
from typing import Annotated
from pydantic import BaseModel, Field, EmailStr
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


# @app.get("/items/")
# async def read_item(skip: int = 0, limit: int = 10):
#     return fake_items_db[skip : skip + limit]


@app.get("/items/{item_id}")
async def read_item(item_id: str, q: str | None = None, short: bool = False):
    item = {"item_id": item_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update(
            {"description": "This is an amazing item that has a long description"}
        )
    return item



class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None



@app.post("/items/")
async def create_item(item: Item):
    item_dict = item.model_dump()
    if item.tax is not None:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict


class UserRegistration(BaseModel):
    username: str = Field(min_length=3, max_length=20)
    age: int = Field(gt=18, description="User must be over 18")
    email: EmailStr

@app.post("/register")
def register(user: UserRegistration):
    # FastAPI automatically validates user.age > 18 and user.email syntax.
    # You receive a strongly typed object ready to use.
    return {"message": f"Welcome {user.username}!"}



def verify_api_key(x_api_key: Annotated[str, Header()]):
    if x_api_key != "my-secret-key":
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

# 2. Inject it into a route handler
@app.get("/secure-data")
def get_secure_data(api_key: Annotated[str, Depends(verify_api_key)]):
    return {"status": "authenticated", "key_used": api_key}


def log_notification(email: str):
    # Imagine writing to a database or sending an email here
    print(f"Notification sent to {email}")

@app.post("/notify")
def send_notification(email: str, background_tasks: BackgroundTasks):
    # Queues the task to run AFTER returning the response to the user
    background_tasks.add_task(log_notification, email)
    return {"message": "Notification scheduled!"}