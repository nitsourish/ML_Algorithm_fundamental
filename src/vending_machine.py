prods = {'coke':100, 'pepsi':150,'coffee':80}
coins = [5,10,20,50,100]

def Vending_machine(prods,coins):
    running_tot = 0
    abort = False
    item = str(input("Enter a item"))
    if item not in prods:
        print('item not in list')
        item = str(input("Enter a item from coke/pepsi/coffee"))
    price = prods[item]
    while running_tot < price and not abort:
        coin = int(input("Enter a coin"))
        if coin not in coins:
             print('coin not valid')
             coin = str(input("Enter a valid coin"))
             print('take your coin back')
        running_tot+=coin     
        response = str(input("would you like to abort Y or N?"))
        if response == 'Y':
            abort = True
            print(f'take your coins total change {running_tot}')
            return
        
    if running_tot >= price:
        print('take you drink')
        change = running_tot-price
        if change > 0:
            print(f'take your change {change}')
        return True             
    else:
        print('take your coins')
        return
