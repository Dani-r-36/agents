# """
# Basic fastapi stuff
# """
# from fastapi import FastAPI, Depends, HTTPException, Header, BackgroundTasks
# from typing import Annotated
# from pydantic import BaseModel, Field, EmailStr
# app = FastAPI()


# @app.get("/")
# async def root():
#     return {"message": "Hello World"}


# fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


# # @app.get("/items/")
# # async def read_item(skip: int = 0, limit: int = 10):
# #     return fake_items_db[skip : skip + limit]


# @app.get("/items/{item_id}")
# async def read_item(item_id: str, q: str | None = None, short: bool = False):
#     item = {"item_id": item_id}
#     if q:
#         item.update({"q": q})
#     if not short:
#         item.update(
#             {"description": "This is an amazing item that has a long description"}
#         )
#     return item



# class Item(BaseModel):
#     name: str
#     description: str | None = None
#     price: float
#     tax: float | None = None



# @app.post("/items/")
# async def create_item(item: Item):
#     item_dict = item.model_dump()
#     if item.tax is not None:
#         price_with_tax = item.price + item.tax
#         item_dict.update({"price_with_tax": price_with_tax})
#     return item_dict


# class UserRegistration(BaseModel):
#     username: str = Field(min_length=3, max_length=20)
#     age: int = Field(gt=18, description="User must be over 18")
#     email: EmailStr

# @app.post("/register")
# def register(user: UserRegistration):
#     # FastAPI automatically validates user.age > 18 and user.email syntax.
#     # You receive a strongly typed object ready to use.
#     return {"message": f"Welcome {user.username}!"}



# def verify_api_key(x_api_key: Annotated[str, Header()]):
#     if x_api_key != "my-secret-key":
#         raise HTTPException(status_code=401, detail="Invalid API Key")
#     return x_api_key

# # 2. Inject it into a route handler
# @app.get("/secure-data")
# def get_secure_data(api_key: Annotated[str, Depends(verify_api_key)]):
#     return {"status": "authenticated", "key_used": api_key}


# def log_notification(email: str):
#     # Imagine writing to a database or sending an email here
#     print(f"Notification sent to {email}")

# @app.post("/notify")
# def send_notification(email: str, background_tasks: BackgroundTasks):
#     # Queues the task to run AFTER returning the response to the user
#     background_tasks.add_task(log_notification, email)
#     return {"message": "Notification scheduled!"}


# """
# sqlachemy for backend storing 
# """

# from typing import AsyncGenerator
# from fastapi import FastAPI, Depends, HTTPException, status
# from contextlib import asynccontextmanager
# from pydantic import BaseModel
# from sqlalchemy import String, select
# from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
# from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# # 1. Database Configuration (Using SQLite + aiosqlite driver)
# DATABASE_URL = "sqlite+aiosqlite:///./test.db"
# engine = create_async_engine(DATABASE_URL, echo=True)
# AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

# # 2. Database Models
# class Base(DeclarativeBase):
#     pass

# class ItemDB(Base):
#     __tablename__ = "items"
    
#     id: Mapped[int] = mapped_column(primary_key=True, index=True)
#     title: Mapped[str] = mapped_column(String, index=True)
#     description: Mapped[str] = mapped_column(String, default="")

# # 3. Dependency to inject session per-request
# async def get_db() -> AsyncGenerator[AsyncSession, None]:
#     async with AsyncSessionLocal() as session:
#         yield session

# # 4. Pydantic Schemas
# class ItemCreate(BaseModel):
#     title: str
#     description: str = ""

# class ItemResponse(ItemCreate):
#     id: int
#     class Config:
#         from_attributes = True

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Create tables automatically on startup
#     async with engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)
#     yield

# app = FastAPI(lifespan=lifespan)
# # 5. Route with Database Dependency
# @app.post("/items/", response_model=ItemResponse, status_code=status.HTTP_201_CREATED)
# async def create_item(item: ItemCreate, db: AsyncSession = Depends(get_db)):
#     db_item = ItemDB(title=item.title, description=item.description)
#     db.add(db_item)
#     await db.commit()
#     await db.refresh(db_item)
#     return db_item

# app = FastAPI(lifespan=lifespan)
# # 5. Route with Database Dependency
# @app.get("/items/{item_id}", response_model=ItemResponse)
# async def get_item(item_id:int, db: AsyncSession = Depends(get_db)):
#     # db_item = ItemDB(id=item_id)
#     # db.add(db_item)
#     # await db.refresh(db_item)
#     # return db_item
#     result = await db.execute(
#         select(ItemDB).where(ItemDB.id == item_id)
#     )
#     item = result.scalar_one_or_none()

#     if item is None:
#         raise HTTPException(
#             status_code=404,
#             detail="Item not found"
#         )

#     return item

"""
websocket with fastapi
"""

from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

class ConnectionManager:
    """Manages active WebSocket connections and message distribution."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    # Notify everyone that a new user joined
    await manager.broadcast(f"System: Client #{client_id} joined the room.")
    
    try:
        while True:
            # Wait for incoming text from this client
            data = await websocket.receive_text()
            
            # Send acknowledgement to sender
            await manager.send_personal_message(f"You: {data}", websocket)
            
            # Broadcast the message to all active connected clients
            await manager.broadcast(f"Client #{client_id}: {data}")
            
    except WebSocketDisconnect:
        # Handle client disconnection gracefully
        manager.disconnect(websocket)
        await manager.broadcast(f"System: Client #{client_id} left the room.")