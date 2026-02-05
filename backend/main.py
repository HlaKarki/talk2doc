from fastapi import FastAPI

app = FastAPI(title="Talk2Doc API")


@app.get("/")
async def root():
    return {"message": "Hello, World!"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
