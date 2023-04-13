# cryptoBandit aka binanceTradingBot
## Simple Python buy-low sell-high trading bot

Python script that interacts with the Binance API to perform trading actions based on certain thresholds set by the user. It begins by establishing a connection with the API and printing out the user's initial USDT balance. It then prompts the user to enter trading variables, such as the trading symbol, the amount of USDT to spend, the buy-low threshold, the sell-high threshold, and the stop-loss threshold. After that, it defines several functions for buying, selling, and getting the available balance of a cryptocurrency. 

The script checks the current percent change in price compared to the buy price, and if it's less than or equal to the stop-loss threshold, it triggers a sell with the sell price set to the buy price minus the stop-loss threshold. This ensures that the script sells the position at a price that's lower than the buy price but still reasonably close to it to minimize losses.

After the sell function is triggered, the script sets the position_is_open variable to False, indicating that there is no longer a position open, and the buy_price and sell_price variables are reset to None, ready for the next buy and sell orders.

It also prints out various information, such as the current price, the percent change, and the profits or losses. The script terminates gracefully upon the user pressing CTRL+C.
