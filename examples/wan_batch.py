import asyncio

from video_gen import VideoClient

PROMPTS = [
    "A snowy forest at dawn",
    "Neon Tokyo street at night",
    "A calm ocean with gentle waves",
]


async def main() -> None:
    async with VideoClient() as client:
        tasks = [
            await client.generate(provider="wan", model="2.1", prompt=prompt, duration=5)
            for prompt in PROMPTS
        ]

        results = await asyncio.gather(*(task.wait() for task in tasks))
        for prompt, result in zip(PROMPTS, results):
            print(f"{prompt} -> {result.video_url}")


if __name__ == "__main__":
    asyncio.run(main())
