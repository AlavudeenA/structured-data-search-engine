-- =========================================
-- Create Database
-- =========================================
CREATE DATABASE Compliance;
GO

USE Compliance;
GO

-- =========================================
-- Employee Table
-- =========================================
CREATE TABLE Employee (
    EmployeeID INT PRIMARY KEY,
    EmployeeName VARCHAR(100) NOT NULL,
    Department VARCHAR(100),
    JobTitle VARCHAR(100),
    HireDate DATE,
    Status VARCHAR(20)
);

-- =========================================
-- Broker Dealer Table
-- =========================================
CREATE TABLE BrokerDealer (
    BrokerDealerID INT PRIMARY KEY,
    BrokerDealerName VARCHAR(200) NOT NULL,
    Country VARCHAR(100),
    RegistrationNumber VARCHAR(100)
);

-- =========================================
-- Account Table
-- =========================================
CREATE TABLE Account (
    AccountID INT PRIMARY KEY,
    EmployeeID INT NOT NULL,
    BrokerDealerID INT NOT NULL,
    AccountNumber VARCHAR(50),
    AccountType VARCHAR(50),
    OpenDate DATE,
    Status VARCHAR(20),

    CONSTRAINT FK_Account_Employee
        FOREIGN KEY (EmployeeID)
        REFERENCES Employee(EmployeeID),

    CONSTRAINT FK_Account_BrokerDealer
        FOREIGN KEY (BrokerDealerID)
        REFERENCES BrokerDealer(BrokerDealerID)
);

-- =========================================
-- Trade Request Table
-- =========================================
CREATE TABLE TradeRequest (
    TradeRequestID INT PRIMARY KEY,
    EmployeeID INT NOT NULL,
    RequestDate DATE,
    SecuritySymbol VARCHAR(20),
    TradeType VARCHAR(10),
    Quantity INT,
    Status VARCHAR(20),

    CONSTRAINT FK_TradeRequest_Employee
        FOREIGN KEY (EmployeeID)
        REFERENCES Employee(EmployeeID)
);