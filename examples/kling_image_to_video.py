import asyncio

from video_gen import VideoClient


async def main() -> None:
    async with VideoClient() as client:
        task = await client.generate(
            provider="kling",
            model="1.6",
            prompt="Slow push-in, subtle wind in the trees",
            image_url="https://example.com/photo.jpg",
            duration=5,
        )
        result = await task.wait()
        print("video:", result.video_url)


if __name__ == "__main__":
    asyncio.run(main())
