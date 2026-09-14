import asyncio

from video_gen import VideoClient


async def main() -> None:
    async with VideoClient() as client:
        task = await client.generate(
            provider="minimax",
            model="hailuo",
            prompt="A drone shot over misty mountains at sunrise",
            duration=6,
        )

        print("submitted:", task.task_id)
        await asyncio.sleep(10)

        status = await task.status()
        print("status:", status.state)

        result = await task.wait()
        print("done:", result.video_url)


if __name__ == "__main__":
    asyncio.run(main())
