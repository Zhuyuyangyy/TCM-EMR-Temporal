"""中医电子病历证候演化时序模型与复诊结局预测"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.predict import router as predict_router
from backend.api.explain import router as explain_router

app = FastAPI(
    title="TCM-EMR-Temporal",
    version="0.2.0",
    description="中医电子病历证候演化时序模型与复诊结局预测",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(predict_router)
app.include_router(explain_router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "TCM-EMR-Temporal", "version": "0.2.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)
