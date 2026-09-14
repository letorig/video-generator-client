import asyncio

from video_gen import VideoClient


async def main() -> None:
    async with VideoClient() as client:
        task = await client.generate(
            provider="seedance",
            model="2.5",
            prompt="A cinematic slow-motion shot of a cyberpunk city in the rain",
            duration=5,
        )
        result = await task.wait(on_update=lambda s: print("state:", s.state))
        print("video:", result.video_url)


if __name__ == "__main__":
    asyncio.run(main())
