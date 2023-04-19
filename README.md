# cryptoBandit aka Binance Trading Bot
## Simple Python buy-low and sell-high trading bot (it also checks if RSI is bellow 30).

**TL;DR: This Python script is a cryptocurrency trading bot that connects to the Binance and Slack APIs. It calculates technical indicators and executes trades based on user-defined parameters. The script is easy to use, includes helper functions, and is the ultimate tool for cryptocurrency trading.**

### Disclaimer
This Python script is a collaborative effort and was written with the assistance of ChatGPT, a language model trained by OpenAI. Please note that the author of this script is not a trained Python developer, so the code should be taken with a grain of salt. The author welcomes suggestions for how to improve the code or make it more efficient.

### Functionality
- This Python script is a cryptocurrency trading bot that connects to the Binance and Slack APIs.
- The script defines variables for the bot version and output file names, and Slack API token and channel name.
- The bot sends a message to the Slack channel when initiated.
- The user is prompted to enter values for various trading parameters such as the symbol to trade and buy and sell thresholds.
- The script also calculates several technical indicators using the Binance API and prints their values to the console, however as of now the bot is using only RSI value to execute buy function. **You are welcome to play with different strategies and experiment with different strategies. Make sure to share the results with us! Bad or good, help the others!**
- The bot executes a buy order if certain conditions are met, and a sell order if the percentage change in price exceeds the sell threshold or drops below the stop loss threshold (or if ten candlesticks have elapsed - note that this function is commented out) .
- The script includes several helper functions such as calculate_profit_or_loss() or reset availability if price rises up.
- It also prints out various information, such as the current price, the percent change, and the profits or losses. 
- The script terminates gracefully upon the user pressing CTRL+C.

### When the purchase is made?
If the conditions are right (i.e., RSI below 30 and price change below the buy threshold), the bot will execute a buy order. But if the percentage change in price exceeds the sell threshold or drops below the stop loss threshold, it's time to sell! After the sell function is triggered, the script sets the position_is_open variable to False, indicating that there is no longer a position open, and the buy_price and sell_price variables are reset to None, ready for the next buy and sell orders.

### What if I want to try different strategies?
Now comes the fun part - technical indicators! The cryptoBandit already calculates several indicators and print their values to the console. We're talking RSI, MACD, EMA, Vortex, Bollinger Bands, and Fibonacci retracements. **Feel free to try them out, and share with us the results to help others in their learning path**. 

### Screens

![image](https://user-images.githubusercontent.com/121772502/233147399-25491da4-b5ef-4c0b-8371-1d624f802e90.png)



