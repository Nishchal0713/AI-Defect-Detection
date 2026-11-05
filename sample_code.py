def calculate_total(items, discount, tax, customer, address, payment):
    total = 0
    if customer:
        if address:
            if payment:
                for item in items:
                    if item > 0:
                        total += item
    return total
