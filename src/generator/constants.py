"""Shared domain constants for the synthetic South African bank."""

PROVINCES = [
    "Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape",
    "Limpopo", "Mpumalanga", "North West", "Free State", "Northern Cape",
]
PROVINCE_WEIGHTS = [0.28, 0.16, 0.19, 0.10, 0.07, 0.07, 0.06, 0.05, 0.02]

OCCUPATIONS = [
    "Professional", "Technician", "Sales", "Teacher", "Healthcare",
    "Public Service", "Student", "Self-Employed", "Operations", "Retired",
]
OCCUPATION_WEIGHTS = [0.18, 0.13, 0.11, 0.09, 0.08, 0.10, 0.07, 0.10, 0.10, 0.04]

MERCHANTS = {
    "Grocery": ["Shoprite", "Pick n Pay", "Checkers", "Woolworths Food", "SPAR"],
    "Fuel": ["Engen", "Shell", "BP", "Sasol", "TotalEnergies"],
    "Retail": ["Mr Price", "PEP", "Game", "Makro", "Takealot"],
    "Dining": ["Nando's", "Spur", "KFC", "Debonairs", "Local Cafe"],
    "Transport": ["Gautrain", "Uber", "Bolt", "Bus Service", "Airline"],
    "Utilities": ["Eskom", "Municipality", "Mobile Network", "Fibre ISP", "Insurance"],
    "Healthcare": ["Pharmacy", "GP Practice", "Hospital", "Dentist", "Optometrist"],
    "Entertainment": ["Netflix", "Showmax", "Cinema", "Gaming Store", "Event Tickets"],
    "Education": ["University", "School Fees", "Bookshop", "Online Course", "Training Provider"],
}

CATEGORY_WEIGHTS = [0.25, 0.13, 0.16, 0.10, 0.08, 0.13, 0.06, 0.05, 0.04]
CHANNELS = ["CARD", "ONLINE", "EFT", "ATM", "MOBILE"]
TRANSACTION_TYPES = ["PURCHASE", "TRANSFER", "CASH_WITHDRAWAL", "DEBIT_ORDER"]
