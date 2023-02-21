# cryptoBandit aka binanceTradingBot
## Simple Python buy-low sell-high trading bot

Introducing cryptoBandit, the notorious cyber-criminal and finance genius! This code is their masterpiece, a ruthless trading bot that can make you rich or leave you destitute, depending on how you set it up. Armed with Python and the Binance API, cryptoBandit can scan the markets for any symbol you choose and buy or sell at a moment's notice, all while taunting you with colorful terminal output.

Enter the parameters carefully, my friend, or you may be sorry. Tell cryptoBandit how much of your precious USDT to spend, how low it should buy before you panic, how high it should sell before you dance a jig, and how much to cut your losses if things go wrong. Once the bot is running, it will watch the symbol like a hawk, looking for the perfect moment to pounce. If it sees a drop that meets your threshold, it will buy in and start the clock. If it sees a rise that meets your threshold, it will sell out and collect its bounty. But beware, dear trader, for cryptoBandit is a wild card. It may decide to bail out early, taking your hard-earned profits and leaving you feeling cheated. Or it may hold on too long, taking your savings down with it as it crashes and burns. But fear not, for you have a secret weapon. If things get too hairy, simply press CTRL+C to kill the bot and escape with what's left of your sanity. It's not much, but it's better than nothing.

So what are you waiting for, my friend? Are you ready to enter the world of high-stakes crypto trading with the infamous cryptoBandit?

### TL;DR

Python script that interacts with the Binance API to perform trading actions based on certain thresholds set by the user. It begins by establishing a connection with the API and printing out the user's initial USDT balance. It then prompts the user to enter trading variables, such as the trading symbol, the amount of USDT to spend, the buy-low threshold, the sell-high threshold, and the stop-loss threshold. After that, it defines several functions for buying, selling, and getting the available balance of a cryptocurrency. 

The script checks the current percent change in price compared to the buy price, and if it's less than or equal to the stop-loss threshold, it triggers a sell with the sell price set to the buy price minus the stop-loss threshold. This ensures that the script sells the position at a price that's lower than the buy price but still reasonably close to it to minimize losses.

After the sell function is triggered, the script sets the position_is_open variable to False, indicating that there is no longer a position open, and the buy_price and sell_price variables are reset to None, ready for the next buy and sell orders.

It also prints out various information, such as the current price, the percent change, and the profits or losses. The script terminates gracefully upon the user pressing CTRL+C.
