import discord
from discord.ext import commands
import chatbot
from config import TOKEN_BOT
import os
import aiohttp

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f'Bot đã hoạt động: {bot.user}')

@bot.command()
async def law(ctx, *, query: str):
    results = chatbot.get_law_answer(query)
    response = "".join(results)
    await ctx.send(response[:2000])  # limit 2000 characters

async def run_bot():
    connector = aiohttp.TCPConnector(ssl=False)  # turn off SSL checking
    async with aiohttp.ClientSession(connector=connector) as session:
        bot.session = session  # session for bot
        await bot.start(TOKEN_BOT, reconnect=True)

import asyncio
asyncio.run(run_bot())
