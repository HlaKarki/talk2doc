"""Test database connection and create tables."""
import asyncio
from database.session import init_db, engine
from sqlalchemy import text


async def test_connection():
    """Test database connection."""
    print("Testing database connection...")
    
    try:
        # Test connection
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT version()"))
            version = result.scalar()
            print(f"✅ Connected to PostgreSQL!")
            print(f"Version: {version}")
        
        # Initialize database (create tables)
        print("\nCreating tables...")
        await init_db()
        print("✅ Tables created successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(test_connection())
