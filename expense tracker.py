import tkinter as tk
from tkinter import messagebox, simpledialog
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import os

# Database setup
DB_FILE = "expenses.db"

class ExpenseTracker:
    def __init__(self):
        self.conn = sqlite3.connect(DB_FILE)
        self.cursor = self.conn.cursor()
        self.create_table()
        self.ml_model = ExpensePredictor()

    def create_table(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS expenses (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                project TEXT,
                                category TEXT,
                                amount REAL,
                                date TEXT)''')
        self.conn.commit()

    def add_expense(self, project, category, amount):
        date = datetime.today().strftime('%Y-%m-%d')
        self.cursor.execute("INSERT INTO expenses (project, category, amount, date) VALUES (?, ?, ?, ?)",
                            (project, category, amount, date))
        self.conn.commit()
        messagebox.showinfo("Success", "Expense added successfully!")

    def get_expenses(self):
        self.cursor.execute("SELECT * FROM expenses")
        return self.cursor.fetchall()

    def plot_expenses(self):
        data = pd.read_sql_query("SELECT date, SUM(amount) as total FROM expenses GROUP BY date", self.conn)
        if data.empty:
            messagebox.showinfo("No Data", "No expenses found!")
            return

        data['date'] = pd.to_datetime(data['date'])
        plt.figure(figsize=(8, 4))
        plt.plot(data['date'], data['total'], marker='o', linestyle='-')
        plt.xlabel("Date")
        plt.ylabel("Total Expense")
        plt.title("Expense Trends")
        plt.grid()
        plt.show()

    def predict_future_expense(self, days_ahead=7):
        return self.ml_model.predict_expenses(days_ahead)

class ExpensePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.train_model()

    def train_model(self):
        if not os.path.exists(DB_FILE):
            return
        
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("SELECT date, amount FROM expenses", conn)
        if df.empty:
            return
        
        df['date'] = pd.to_datetime(df['date']).map(datetime.toordinal)
        X = df[['date']]
        y = df['amount']

        self.model.fit(X, y)

    def predict_expenses(self, days_ahead=7):
        future_dates = [datetime.today() + timedelta(days=i) for i in range(1, days_ahead+1)]
        future_ordinal = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)

        predictions = self.model.predict(future_ordinal)
        return list(zip(future_dates, predictions))

class ExpenseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Expense Tracker")
        self.tracker = ExpenseTracker()

        tk.Label(root, text="Project:").pack()
        self.project_entry = tk.Entry(root)
        self.project_entry.pack()

        tk.Label(root, text="Category:").pack()
        self.category_entry = tk.Entry(root)
        self.category_entry.pack()

        tk.Label(root, text="Amount:").pack()
        self.amount_entry = tk.Entry(root)
        self.amount_entry.pack()

        tk.Button(root, text="Add Expense", command=self.add_expense).pack(pady=5)
        tk.Button(root, text="Show Expenses", command=self.show_expenses).pack()
        tk.Button(root, text="Show Expense Trends", command=self.tracker.plot_expenses).pack(pady=5)
        tk.Button(root, text="Predict Future Expenses", command=self.predict_expenses).pack()

    def add_expense(self):
        project = self.project_entry.get()
        category = self.category_entry.get()
        amount = self.amount_entry.get()

        if not project or not category or not amount.isdigit():
            messagebox.showerror("Error", "Please enter valid data!")
            return

        self.tracker.add_expense(project, category, float(amount))

    def show_expenses(self):
        expenses = self.tracker.get_expenses()
        if not expenses:
            messagebox.showinfo("No Data", "No expenses recorded!")
            return
        
        result = "\n".join([f"{exp[1]} | {exp[2]} | ${exp[3]} | {exp[4]}" for exp in expenses])
        messagebox.showinfo("Expenses", result)

    def predict_expenses(self):
        predictions = self.tracker.predict_future_expense()
        if not predictions:
            messagebox.showinfo("No Data", "Not enough data for prediction!")
            return

        result = "\n".join([f"{date.strftime('%Y-%m-%d')}: ${amount:.2f}" for date, amount in predictions])
        messagebox.showinfo("Predicted Expenses", result)

if __name__ == "__main__":
    root = tk.Tk()
    app = ExpenseApp(root)
    root.mainloop()
