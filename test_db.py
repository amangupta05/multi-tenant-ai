import asyncio
from sqlalchemy import select
from app.db.database import _get_session_factory, init_db
from app.db.models import Document

async def main():
    await init_db()
    factory = _get_session_factory()
    async with factory() as session:
        result = await session.execute(select(Document).order_by(Document.created_at.desc()).limit(3))
        docs = result.scalars().all()
        for d in docs:
            print(f"ID: {d.id} | NAME: {d.filename} | STATUS: {d.status} | MD: {d.doc_metadata}")

asyncio.run(main())
