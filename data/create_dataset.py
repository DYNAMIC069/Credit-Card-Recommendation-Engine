import pandas as pd
import numpy as np
import os

# Always resolve paths relative to THIS file — works on Windows, Mac, Linux
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = HERE  # this file lives inside data/

cards = [
    # bank, name, network, jf, af, rt, br, brt, brc, ld, li, fx, mi, seg, rating, bf, lf
    ("HDFC Bank","Millennia Credit Card","Mastercard",1000,1000,"Cashback",1.0,5.0,"Online Shopping",True,False,3.5,6,"Mid",4.8,"Shopping",False),
    ("HDFC Bank","Regalia Gold Credit Card","Mastercard",2500,2500,"Reward Points",1.73,8.6,"Myntra/Nykaa/M&S",True,True,2.0,8,"Mid-Premium",4.5,"Travel|Shopping",False),
    ("HDFC Bank","Swiggy BLCK Credit Card","Mastercard",1000,1000,"Cashback",1.0,10.0,"Swiggy App",False,False,3.5,5,"Entry",4.6,"Food|Dining",False),
    ("HDFC Bank","Tata Neu Infinity Credit Card","RuPay",1499,1499,"NeuCoins",1.5,10.0,"Tata Neu App",True,True,2.0,6,"Mid",4.5,"Travel|Shopping",False),
    ("HDFC Bank","Tata Neu Plus Credit Card","RuPay",499,499,"NeuCoins",1.0,7.0,"Tata Neu App",True,False,3.5,3,"Entry",4.5,"Shopping",False),
    ("HDFC Bank","PhonePe HDFC Ultimo","RuPay",999,999,"Reward Points",1.0,10.0,"PhonePe recharges",True,False,3.5,4,"Mid",4.6,"Utility|Shopping|UPI",False),
    ("HDFC Bank","PhonePe HDFC Uno","RuPay",499,499,"Reward Points",1.0,2.0,"PhonePe spends",False,False,3.5,3,"Entry",4.5,"Shopping|Utility",False),
    ("HDFC Bank","Swiggy Ornge Credit Card","Mastercard",500,500,"Cashback",1.0,5.0,"Swiggy App",False,False,3.5,3,"Entry",4.4,"Food|Shopping",False),
    ("HDFC Bank","RuPay IRCTC Credit Card","RuPay",500,500,"Reward Points",1.0,5.0,"IRCTC bookings",False,False,3.5,3,"Entry",4.5,"Travel",False),
    ("HDFC Bank","Shoppers Stop Credit Card","Mastercard",299,299,"Reward Points",1.0,3.0,"Shoppers Stop",False,False,3.5,3,"Entry",4.5,"Shopping",False),
    ("HDFC Bank","MoneyBack Plus Credit Card","Mastercard",500,500,"Reward Points",2.0,10.0,"Flipkart/Amazon/Swiggy",False,False,3.5,3,"Entry",4.5,"Shopping",False),
    ("HDFC Bank","Marriott Bonvoy Credit Card","Mastercard",3000,3000,"Reward Points",2.0,8.0,"Marriott Hotels",True,True,2.0,8,"Mid-Premium",4.6,"Travel|Stays",False),
    ("HDFC Bank","Diners Club Black Metal Edition","Diners Club",10000,10000,"Reward Points",3.33,33.33,"SmartBuy 10X",True,True,2.0,20,"Super Premium",4.7,"Travel|Dining|Shopping",False),
    ("HDFC Bank","INFINIA Metal Credit Card","Mastercard",12500,12500,"Reward Points",3.33,33.33,"SmartBuy 10X",True,True,2.0,100,"Super Premium",5.0,"Travel|Dining|Shopping",False),
    ("HDFC Bank","Diners Club Privilege Card","Diners Club",1000,1000,"Reward Points",1.33,6.67,"Swiggy & Zomato",True,True,2.0,6,"Mid",4.5,"Movies|Travel|Food",False),
    ("HDFC Bank","Shoppers Stop Black Credit Card","Mastercard",4500,4500,"Reward Points",2.0,7.0,"Shoppers Stop",True,True,3.5,12,"Premium",4.5,"Travel|Shopping",False),
    ("HDFC Bank","Freedom Credit Card","Mastercard",500,500,"Reward Points",1.0,10.0,"BigBasket/Swiggy/OYO",False,False,3.5,3,"Entry",4.2,"Shopping",False),
    ("HDFC Bank","IndianOil HDFC Credit Card","Mastercard",500,500,"Fuel Points",1.0,5.0,"IndianOil outlets",False,False,3.5,3,"Entry",4.5,"Fuel",False),
    ("HDFC Bank","Swiggy HDFC Credit Card","Mastercard",500,500,"Cashback",1.0,10.0,"Swiggy App",False,False,3.5,3,"Entry",4.8,"Food|Dining",False),
    ("HDFC Bank","UPI RuPay Credit Card","RuPay",99,99,"Reward Points",1.0,3.0,"Grocery/Dining",False,False,3.5,2,"Entry",4.5,"Shopping",False),
    ("HDFC Bank","Diners Club Black Credit Card","Diners Club",10000,10000,"Reward Points",3.33,33.33,"SmartBuy 10X",True,True,2.0,20,"Super Premium",4.5,"Travel|Dining|Shopping",False),
    ("HDFC Bank","Pixel Play Credit Card","Mastercard",0,0,"Cashback",1.0,5.0,"Two chosen categories",False,False,3.5,3,"Entry",4.5,"Shopping",True),
    ("HDFC Bank","Pixel Go Credit Card","Mastercard",250,250,"Cashback",1.0,1.0,"All spends",False,False,3.5,3,"Entry",4.4,"Shopping",False),
    ("SBI Card","Cashback SBI Credit Card","Mastercard",999,999,"Cashback",1.0,5.0,"All online spends",False,False,3.5,5,"Entry-Mid",4.7,"Shopping|Food",False),
    ("SBI Card","SimplyCLICK Credit Card","Mastercard",499,499,"Reward Points",1.0,10.0,"Apollo/Swiggy/BookMyShow",False,False,3.5,3,"Entry",4.8,"Shopping",False),
    ("SBI Card","PhonePe SBI PURPLE","RuPay",499,499,"Reward Points",1.0,3.0,"PhonePe spends",False,False,3.5,3,"Entry",4.6,"Shopping|Utility",False),
    ("SBI Card","Flipkart SBI Credit Card","Mastercard",500,500,"Cashback",1.0,7.5,"Flipkart/Myntra",False,False,3.5,3,"Entry",4.8,"Shopping",False),
    ("SBI Card","Tata Neu Plus SBI Credit Card","RuPay",499,499,"NeuCoins",1.0,2.0,"Tata Neu + brands",True,False,3.5,3,"Entry",4.4,"Travel|Shopping",False),
    ("SBI Card","BPCL SBI Credit Card","Mastercard",499,499,"Reward Points",0.25,3.25,"BPCL fuel stations",False,False,3.5,3,"Entry",4.6,"Fuel",False),
    ("SBI Card","PhonePe SBI SELECT BLACK","Mastercard",1499,1499,"Reward Points",1.0,10.0,"PhonePe spends",True,True,3.5,8,"Mid",4.8,"Shopping|Utility",False),
    ("SBI Card","Tata Neu Infinity SBI Credit Card","RuPay",1499,1499,"NeuCoins",1.5,5.0,"Tata Neu + brands",True,False,1.99,6,"Mid",4.5,"Travel|Shopping",False),
    ("SBI Card","BPCL SBI Card OCTANE","Mastercard",1499,1499,"Reward Points",0.25,7.25,"BPCL fuel stations",True,False,3.5,8,"Mid",4.7,"Travel|Fuel|Food",False),
    ("SBI Card","SBI Card PULSE","Visa",1499,1499,"Reward Points",0.5,2.5,"Movies/Dining/Pharmacy",True,False,3.5,6,"Mid",4.5,"Travel|Shopping|Health",False),
    ("SBI Card","SimplySAVE UPI RuPay","RuPay",499,499,"Reward Points",0.17,1.7,"Movies/Grocery/Dining",False,False,3.5,3,"Entry",4.6,"Shopping",False),
    ("SBI Card","IndiGo SBI Credit Card","Mastercard",1499,1499,"Air Miles",0.25,0.75,"IndiGo app/web",True,False,3.5,6,"Mid",4.5,"Travel",False),
    ("SBI Card","IndiGo SBI ELITE Credit Card","Mastercard",4999,4999,"Air Miles",0.5,1.75,"IndiGo app/web",True,True,3.5,12,"Premium",4.5,"Travel",False),
    ("SBI Card","Apollo SBI SELECT Credit Card","Mastercard",1499,1499,"Reward Points",0.5,10.0,"Apollo 24|7 App",True,False,3.5,6,"Mid",4.6,"Travel|Shopping|Health",False),
    ("SBI Card","IRCTC SBI RuPay Credit Card","RuPay",500,500,"Reward Points",0.8,10.0,"IRCTC train bookings",False,False,3.5,3,"Entry",4.3,"Travel",False),
    ("SBI Card","Reliance SBI Card","Visa",499,499,"Reward Points",0.25,1.25,"Reliance Retail stores",False,False,3.5,3,"Entry",4.3,"Shopping",False),
    ("SBI Card","IRCTC SBI Card Premier","Mastercard",1499,1499,"Reward Points",0.8,2.4,"Dining & utility",False,False,3.5,6,"Mid",4.3,"Travel",False),
    ("SBI Card","SBI MILES Credit Card","Mastercard",1499,1499,"Travel Credits",0.5,1.0,"Travel spends",True,False,3.5,6,"Mid",4.4,"Travel",False),
    ("SBI Card","SBI Prime Credit Card","Mastercard",2999,2999,"Reward Points",0.5,2.5,"Dining/Grocery/Dept",True,True,3.5,10,"Premium",4.4,"Travel|Shopping",False),
    ("SBI Card","Titan SBI Credit Card","Mastercard",2999,2999,"Reward Points",1.5,7.5,"Titan brand stores",True,True,3.5,10,"Premium",4.4,"Shopping",False),
    ("SBI Card","Landmark Rewards SBI PRIME","Visa",2999,2999,"Reward Points",0.5,6.25,"Landmark stores",True,True,3.5,10,"Premium",4.5,"Travel|Shopping",False),
    ("SBI Card","Landmark Rewards SBI SELECT","Visa",1499,1499,"Reward Points",0.5,3.75,"Landmark stores",True,False,3.5,6,"Mid",4.5,"Shopping",False),
    ("SBI Card","Landmark Rewards SBI Card","Visa",499,499,"Reward Points",0.25,2.5,"Landmark stores",False,False,3.5,3,"Entry",4.4,"Shopping",False),
    ("SBI Card","SBI AURUM Credit Card","Mastercard",9999,9999,"Reward Points",1.0,4.0,"All eligible spends",True,True,1.99,20,"Super Premium",4.2,"Movies|Travel|Dining",False),
    ("SBI Card","Reliance SBI Card PRIME","Visa",2999,2999,"Reward Points",0.5,2.5,"Reliance stores",True,True,3.5,10,"Premium",4.6,"Shopping",False),
    ("SBI Card","KrisFlyer SBI Credit Card","Mastercard",2999,2999,"KrisFlyer Miles",0.5,2.5,"Singapore Airlines",True,True,3.5,10,"Premium",4.5,"Travel",False),
    ("SBI Card","SBI ELITE Credit Card","Mastercard",4999,4999,"Reward Points",0.5,2.5,"Dining/Grocery/Dept",True,True,1.99,15,"Premium",4.4,"Movies|Travel|Shopping",False),
    ("SBI Card","SBI MILES PRIME Credit Card","Mastercard",2999,2999,"Travel Credits",1.0,2.0,"Travel spends",True,True,3.5,10,"Premium",4.5,"Travel",False),
    ("SBI Card","SBI Miles Elite Credit Card","Mastercard",4999,4999,"Travel Credits",1.0,3.0,"Travel spends",True,True,3.5,15,"Premium",4.5,"Travel",False),
    ("SBI Card","KrisFlyer SBI Card Apex","Mastercard",9999,9999,"KrisFlyer Miles",2.0,5.0,"Singapore Airlines",True,True,3.5,20,"Super Premium",4.6,"Travel",False),
    ("SBI Card","SBI Shaurya Credit Card","Mastercard",250,250,"Reward Points",0.25,1.25,"Grocery/Dining/CSD",False,False,3.5,2,"Entry",4.5,"Shopping",False),
    ("SBI Card","SBI Shaurya Select Credit Card","Mastercard",0,1499,"Reward Points",0.5,2.5,"Dining/Grocery/Movies",True,False,3.5,6,"Mid",4.5,"Travel|Shopping",False),
    ("SBI Card","Doctor's IMA SBI Card","Mastercard",1499,1499,"Reward Points",0.25,1.25,"Medical supplies",True,False,3.5,8,"Mid",4.5,"Travel",False),
    ("Axis Bank","Flipkart Axis Bank Credit Card","Visa",0,0,"Cashback",1.0,7.5,"Flipkart/Myntra",False,False,3.5,3,"Entry",4.6,"Shopping",True),
    ("Axis Bank","Horizon Credit Card","Mastercard",3000,3000,"EDGE Miles",2.0,5.0,"Travel spends",True,True,2.0,10,"Mid-Premium",4.7,"Travel",False),
    ("Axis Bank","Airtel Axis Bank Credit Card","Visa",500,500,"Cashback",1.0,25.0,"Airtel recharges",False,False,3.5,3,"Entry",4.8,"Utility|Food",False),
    ("Axis Bank","IndianOil Axis Bank Premium","Mastercard",1000,1000,"EDGE Miles",1.0,4.0,"IndianOil outlets",False,False,3.5,4,"Entry-Mid",4.5,"Fuel",False),
    ("Axis Bank","Axis Bank Cashback Credit Card","Visa",1000,1000,"Cashback",0.5,7.0,"Online purchases",False,False,3.5,4,"Entry-Mid",4.5,"Shopping",False),
    ("Axis Bank","Magnus Credit Card","Mastercard",10000,10000,"EDGE Miles",3.33,10.0,"All spends 10X",True,True,2.0,25,"Super Premium",4.8,"Travel|Shopping|Dining",False),
    ("Axis Bank","Reserve Credit Card","Mastercard",50000,50000,"EDGE Miles",3.33,10.0,"All eligible spends",True,True,1.5,100,"Super Premium",4.7,"Travel|Dining|Luxury",False),
    ("Axis Bank","Neo Credit Card","Mastercard",250,250,"Reward Points",0.5,2.0,"Online shopping",False,False,3.5,3,"Entry",4.3,"Shopping",False),
    ("Axis Bank","Ace Credit Card","Visa",499,499,"Cashback",1.0,5.0,"Google Pay bills",False,False,3.5,3,"Entry",4.6,"Utility|Shopping",False),
    ("Axis Bank","MY Zone Credit Card","Mastercard",500,500,"Reward Points",1.0,5.0,"Zomato + movies",False,False,3.5,3,"Entry",4.4,"Food|Entertainment",False),
    ("Axis Bank","Air India Axis Bank Infinite","Mastercard",10000,10000,"Air Miles",1.0,4.0,"Air India bookings",True,True,2.0,20,"Premium",4.5,"Travel",False),
    ("Axis Bank","Privilege Credit Card","Mastercard",1500,1500,"EDGE Miles",2.0,5.0,"All eligible spends",True,True,2.0,10,"Mid",4.4,"Travel|Shopping",False),
    ("ICICI Bank","Amazon Pay ICICI Bank Credit Card","Visa",0,0,"Cashback",1.0,5.0,"Amazon Prime",False,False,3.5,3,"Entry",5.0,"Shopping",True),
    ("ICICI Bank","Emeralde Private Metal Credit Card","Visa",12499,12499,"Reward Points",3.0,6.0,"All eligible spends",True,True,2.0,100,"Super Premium",4.6,"Travel|Shopping",False),
    ("ICICI Bank","Times Black Credit Card","Mastercard",20000,20000,"Reward Points",2.0,2.5,"International spends",True,True,2.0,100,"Super Premium",4.5,"Travel|Shopping",False),
    ("ICICI Bank","Coral Credit Card","Visa",500,500,"Reward Points",2.0,4.0,"All eligible spends",True,False,3.5,5,"Entry-Mid",4.3,"Shopping|Movies",False),
    ("ICICI Bank","Sapphiro Credit Card","Visa",6500,6500,"Reward Points",2.0,6.0,"All eligible spends",True,True,2.5,15,"Premium",4.4,"Travel|Dining|Shopping",False),
    ("ICICI Bank","Rubyx Credit Card","Mastercard",2000,2000,"Reward Points",2.0,4.0,"All eligible spends",True,False,3.5,8,"Mid",4.3,"Travel|Shopping",False),
    ("ICICI Bank","MakeMyTrip ICICI Signature","Mastercard",2500,2500,"Reward Points",1.5,4.0,"MakeMyTrip bookings",True,True,3.5,8,"Mid-Premium",4.4,"Travel",False),
    ("IDFC FIRST Bank","FIRST Classic Credit Card","Visa",0,0,"Reward Points",1.5,5.0,"Spends above 20K/mo",False,False,3.5,3,"Entry",4.5,"Shopping|Travel",True),
    ("IDFC FIRST Bank","FIRST Select Credit Card","Visa",0,0,"Reward Points",1.5,5.0,"Spends above 20K/mo",True,False,1.5,5,"Mid",4.4,"Travel|Shopping",True),
    ("IDFC FIRST Bank","FIRST Wealth Credit Card","Mastercard",0,0,"Reward Points",1.5,5.0,"Spends above 20K/mo",True,True,1.5,12,"Mid-Premium",4.5,"Travel|Shopping",True),
    ("IDFC FIRST Bank","Ashva Metal Credit Card","Mastercard",2999,2999,"Reward Points",1.0,6.67,"Spends above 20K (10X)",True,True,1.0,12,"Premium",4.7,"Travel|Shopping",False),
    ("IDFC FIRST Bank","Mayura Credit Card","Mastercard",5999,5999,"Reward Points",1.0,6.67,"Spends above 20K (10X)",True,True,1.0,20,"Super Premium",4.8,"Movies|Travel|Shopping",False),
    ("IDFC FIRST Bank","HPCL Power Plus Credit Card","Visa",0,0,"Reward Points",1.0,6.5,"HPCL fuel stations",False,False,3.5,3,"Entry",4.4,"Fuel",True),
    ("AU Small Finance Bank","LIT Credit Card","Visa",0,0,"Reward Points",1.0,10.0,"Paid categories",True,False,3.5,3,"Entry",4.5,"Shopping",True),
    ("AU Small Finance Bank","Altura Credit Card","Visa",0,0,"Reward Points",1.0,2.0,"All spends",False,False,3.5,3,"Entry",4.3,"Shopping",True),
    ("AU Small Finance Bank","Altura Plus Credit Card","Visa",999,999,"Reward Points",1.5,5.0,"All spends",True,False,3.5,5,"Entry-Mid",4.4,"Travel|Shopping",False),
    ("AU Small Finance Bank","Xcite Credit Card","Visa",4999,4999,"Reward Points",3.33,10.0,"Travel spends",True,True,2.0,15,"Premium",4.5,"Travel|Shopping",False),
    ("AU Small Finance Bank","Vetta Credit Card","Visa",2999,2999,"Reward Points",2.0,5.0,"All eligible spends",True,True,2.0,10,"Mid-Premium",4.4,"Travel|Shopping",False),
    ("Federal Bank / BOBCARD","Scapia Credit Card","Visa",0,0,"Scapia Coins",2.0,20.0,"Travel bookings",True,False,0.0,3,"Entry-Mid",4.7,"Travel|Shopping",True),
    ("IndusInd Bank","EazyDiner IndusInd Platinum","Mastercard",0,0,"Reward Points",2.0,4.0,"Dining spends",False,False,3.5,3,"Entry",4.5,"Dining|Food",True),
    ("IndusInd Bank","Legend Credit Card","Visa",9999,9999,"Reward Points",3.0,5.0,"All eligible spends",True,True,1.8,20,"Super Premium",4.5,"Travel|Dining|Shopping",False),
    ("IndusInd Bank","Pinnacle Credit Card","Visa",2999,2999,"Reward Points",2.0,6.0,"All eligible spends",True,True,2.0,10,"Premium",4.4,"Travel|Shopping",False),
    ("IndusInd Bank","Platinum Credit Card","Visa",0,0,"Reward Points",1.0,3.0,"All eligible spends",False,False,3.5,3,"Entry",4.2,"Shopping",True),
    ("American Express","Membership Rewards Credit Card","Amex",1500,4500,"Membership Rewards",0.5,5.0,"All spends 5X SmartEarn",False,False,3.5,6,"Mid",4.6,"Shopping|Dining",False),
    ("American Express","Platinum Travel Credit Card","Amex",3500,5000,"Membership Rewards",1.0,5.0,"Travel bookings 5X",True,False,3.5,8,"Mid-Premium",4.6,"Travel",False),
    ("American Express","Gold Credit Card","Amex",1000,4999,"Membership Rewards",1.0,5.0,"All spends 5X SmartEarn",False,False,3.5,5,"Mid",4.4,"Shopping|Dining",False),
    ("American Express","Platinum Charge Card","Amex",60000,60000,"Membership Rewards",1.0,5.0,"All eligible spends",True,True,2.0,100,"Super Premium",4.8,"Travel|Luxury|Dining",False),
    ("HSBC Bank","Live+ Credit Card","Visa",999,999,"Cashback",1.5,10.0,"Dining & lifestyle",True,False,3.5,5,"Entry-Mid",4.5,"Dining|Food|Shopping",False),
    ("HSBC Bank","Premier Credit Card","Mastercard",0,0,"Reward Points",1.0,5.0,"All eligible spends",True,True,2.0,20,"Premium",4.4,"Travel|Shopping",True),
    ("Kotak Mahindra Bank","811 Dream Credit Card","Mastercard",0,0,"Reward Points",1.0,2.0,"All eligible spends",False,False,3.5,3,"Entry",4.2,"Shopping",True),
    ("Kotak Mahindra Bank","League Platinum Credit Card","Visa",499,499,"Reward Points",1.0,4.0,"All eligible spends",False,False,3.5,3,"Entry",4.3,"Shopping",False),
    ("Kotak Mahindra Bank","White Reserve Credit Card","Visa",3000,3000,"Reward Points",1.0,6.0,"All eligible spends",True,True,2.0,10,"Premium",4.4,"Travel|Lifestyle",False),
    ("Standard Chartered","Manhattan Platinum Card","Mastercard",999,999,"Cashback",0.5,5.0,"Supermarkets/dining",False,False,3.5,4,"Entry",4.3,"Shopping|Dining",False),
    ("Standard Chartered","Smart Credit Card","Mastercard",0,499,"Cashback",1.0,5.0,"Online spends",True,False,3.5,5,"Entry",4.4,"Shopping",False),
    ("Yes Bank","YES FIRST Preferred Credit Card","Mastercard",2999,2999,"Reward Points",2.0,8.0,"All eligible spends",True,True,2.0,10,"Mid-Premium",4.4,"Travel|Shopping",False),
    ("Yes Bank","YES FIRST Exclusive Credit Card","Mastercard",9999,9999,"Reward Points",3.0,12.0,"All eligible spends",True,True,1.75,20,"Super Premium",4.5,"Travel|Luxury",False),
    ("RBL Bank","Shoprite Credit Card","Mastercard",500,500,"Cashback",0.5,5.0,"Groceries/online",False,False,3.5,3,"Entry",4.2,"Shopping|Grocery",False),
    ("Federal Bank","Signet Credit Card","Visa",0,499,"Reward Points",1.0,3.0,"All eligible spends",False,False,3.5,3,"Entry",4.2,"Shopping",False),
    ("Bank of Baroda","Eterna Credit Card","Mastercard",2499,2499,"Reward Points",1.0,6.0,"All eligible spends",True,True,2.0,8,"Mid-Premium",4.3,"Travel|Shopping",False),
    ("Bank of Baroda","Premier Credit Card","Visa",999,999,"Reward Points",1.0,3.0,"All eligible spends",True,False,3.5,5,"Entry-Mid",4.2,"Shopping",False),
    ("PNB","Pride Platinum Credit Card","RuPay",500,500,"Reward Points",1.0,3.0,"All spends",False,False,3.5,3,"Entry",4.0,"Shopping",False),
    ("Union Bank","RuPay Select Credit Card","RuPay",0,0,"Reward Points",1.0,2.0,"All spends",False,False,3.5,3,"Entry",4.0,"Shopping",True),
    ("Canara Bank","Platinum Credit Card","Visa",500,500,"Reward Points",1.0,2.0,"All spends",False,False,3.5,3,"Entry",3.9,"Shopping",False),
]

cols = ["bank","name","network","joining_fee","annual_fee","reward_type","base_reward_rate","best_reward_rate","best_category","lounge_domestic","lounge_international","forex_markup","min_income_lpa","segment","rating","best_for","lifetime_free"]
df = pd.DataFrame(cards, columns=cols)

# Derived features
df['is_cashback'] = df['reward_type'].str.lower().str.contains('cashback').astype(int)
df['is_miles'] = df['reward_type'].str.lower().str.contains('miles|coins').astype(int)
df['has_both_lounge'] = (df['lounge_domestic'] & df['lounge_international']).astype(int)
df['is_lifetime_free'] = df['lifetime_free'].astype(int)
df['forex_low'] = (df['forex_markup'] <= 2.0).astype(int)
df['fee_tier'] = pd.cut(df['annual_fee'], bins=[-1,0,500,1500,5000,100000],
                         labels=['Free','Entry','Mid','Premium','Super Premium'])
df['segment_encoded'] = df['segment'].map({
    'Entry':0,'Entry-Mid':1,'Mid':2,'Mid-Premium':3,'Premium':4,'Super Premium':5})

out_path = os.path.join(DATA_DIR, 'cards.csv')
df.to_csv(out_path, index=False)
print(f"Dataset saved: {len(df)} cards, {len(df.columns)} features")
print(f"Saved to: {out_path}")
print(df[['bank','name','annual_fee','best_reward_rate','rating']].head())
