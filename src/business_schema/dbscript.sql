-- =============================================
-- COMPLIANCE DATABASE - SQLite DDL + SAMPLE DATA
-- Tables: Employee, BrokerDealer, Account,
--         TradeRequest, ComplianceAlert,
--         RestrictedSecurity, ApprovalWorkflow
-- =============================================

PRAGMA foreign_keys = ON;

-- =============================================
-- DROP existing tables (safe order)
-- =============================================
DROP TABLE IF EXISTS ApprovalWorkflow;
DROP TABLE IF EXISTS ComplianceAlert;
DROP TABLE IF EXISTS TradeRequest;
DROP TABLE IF EXISTS RestrictedSecurity;
DROP TABLE IF EXISTS Account;
DROP TABLE IF EXISTS BrokerDealer;
DROP TABLE IF EXISTS Employee;

-- =============================================
-- 1. EMPLOYEE
-- =============================================
CREATE TABLE Employee (
    EmployeeID   INTEGER NOT NULL PRIMARY KEY,
    EmployeeName TEXT    NOT NULL,
    Department   TEXT,
    JobTitle     TEXT,
    HireDate     TEXT,
    Status       TEXT,   -- Active, Terminated
    UpdatedAt    TEXT    DEFAULT (datetime('now'))
);

-- =============================================
-- 2. BROKERDEALER
-- =============================================
CREATE TABLE BrokerDealer (
    BrokerDealerID     INTEGER NOT NULL PRIMARY KEY,
    BrokerDealerName   TEXT    NOT NULL,
    Country            TEXT,
    RegistrationNumber TEXT,
    UpdatedAt          TEXT    DEFAULT (datetime('now'))
);

-- =============================================
-- 3. ACCOUNT
-- =============================================
CREATE TABLE Account (
    AccountID      INTEGER NOT NULL PRIMARY KEY,
    EmployeeID     INTEGER NOT NULL REFERENCES Employee(EmployeeID),
    BrokerDealerID INTEGER NOT NULL REFERENCES BrokerDealer(BrokerDealerID),
    AccountNumber  TEXT,
    AccountType    TEXT,
    OpenDate       TEXT,
    Status         TEXT,   -- Active, Closed
    UpdatedAt      TEXT    DEFAULT (datetime('now'))
);

-- =============================================
-- 4. RESTRICTEDSECURITY
-- =============================================
CREATE TABLE RestrictedSecurity (
    RestrictionID   INTEGER NOT NULL PRIMARY KEY,
    SecuritySymbol  TEXT    NOT NULL,
    RestrictionType TEXT,   -- Blackout, Insider List, Watch List
    StartDate       TEXT,
    EndDate         TEXT,   -- NULL = still active
    Reason          TEXT,
    AddedBy         TEXT,
    UpdatedAt       TEXT    DEFAULT (datetime('now'))
);

-- =============================================
-- 5. TRADEREQUEST
-- =============================================
CREATE TABLE TradeRequest (
    TradeRequestID INTEGER NOT NULL PRIMARY KEY,
    EmployeeID     INTEGER NOT NULL REFERENCES Employee(EmployeeID),
    BrokerDealerID INTEGER NOT NULL REFERENCES BrokerDealer(BrokerDealerID),
    RequestDate    TEXT,
    SecuritySymbol TEXT,
    TradeType      TEXT,   -- BUY, SELL
    Quantity       INTEGER,
    Status         TEXT,   -- Pending, Approved, Rejected, Escalated
    UpdatedAt      TEXT    DEFAULT (datetime('now'))
);

-- =============================================
-- 6. COMPLIANCEALERT
-- =============================================
CREATE TABLE ComplianceAlert (
    AlertID        INTEGER NOT NULL PRIMARY KEY,
    TradeRequestID INTEGER REFERENCES TradeRequest(TradeRequestID),
    EmployeeID     INTEGER NOT NULL REFERENCES Employee(EmployeeID),
    AlertType      TEXT,   -- Insider Trading, Excessive Volume, Restricted Security, Unusual Pattern
    AlertDate      TEXT,
    Severity       TEXT,   -- Low, Medium, High, Critical
    Status         TEXT,   -- Open, Investigating, Closed, Escalated
    Description    TEXT,
    ResolvedDate   TEXT,
    UpdatedAt      TEXT    DEFAULT (datetime('now'))
);

-- =============================================
-- 7. APPROVALWORKFLOW
-- =============================================
CREATE TABLE ApprovalWorkflow (
    WorkflowID     INTEGER NOT NULL PRIMARY KEY,
    TradeRequestID INTEGER NOT NULL REFERENCES TradeRequest(TradeRequestID),
    ReviewerID     INTEGER NOT NULL REFERENCES Employee(EmployeeID),
    ReviewDate     TEXT,
    Decision       TEXT,   -- Approved, Rejected, Escalated, Pending
    Comments       TEXT,
    TurnaroundDays INTEGER, -- ReviewDate - RequestDate
    UpdatedAt      TEXT    DEFAULT (datetime('now'))
);


-- =============================================
-- SAMPLE DATA
-- =============================================

-- --------------------------------------------
-- EMPLOYEE  (20 employees, mixed departments)
-- --------------------------------------------
INSERT INTO Employee (EmployeeID, EmployeeName, Department, JobTitle, HireDate, Status) VALUES
(1,  'John Smith',      'Investment Banking', 'Analyst',           '2021-01-10', 'Active'),
(2,  'Sarah Johnson',   'Compliance',         'Manager',           '2020-03-11', 'Active'),
(3,  'David Lee',       'Technology',         'Engineer',          '2019-04-01', 'Active'),
(4,  'Emily Davis',     'Investment Banking', 'Senior Analyst',    '2018-06-15', 'Active'),
(5,  'Michael Brown',   'Trading',            'Trader',            '2017-09-20', 'Active'),
(6,  'Jessica Wilson',  'Compliance',         'Analyst',           '2022-02-01', 'Active'),
(7,  'James Taylor',    'Investment Banking', 'VP',                '2016-03-10', 'Active'),
(8,  'Amanda Martinez', 'Trading',            'Senior Trader',     '2015-07-05', 'Active'),
(9,  'Robert Anderson', 'Risk Management',    'Risk Analyst',      '2020-11-20', 'Active'),
(10, 'Linda Thomas',    'Compliance',         'Director',          '2014-01-15', 'Active'),
(11, 'Kevin Jackson',   'Investment Banking', 'Analyst',           '2023-03-01', 'Active'),
(12, 'Patricia White',  'Trading',            'Trader',            '2021-08-10', 'Active'),
(13, 'Christopher Harris','Technology',       'Senior Engineer',   '2018-12-01', 'Active'),
(14, 'Barbara Clark',   'Risk Management',    'Senior Risk Analyst','2019-05-20','Active'),
(15, 'Daniel Lewis',    'Investment Banking', 'Associate',         '2022-07-15', 'Active'),
(16, 'Susan Robinson',  'Compliance',         'Analyst',           '2021-09-01', 'Active'),
(17, 'Paul Walker',     'Trading',            'Trader',            '2020-04-15', 'Active'),
(18, 'Nancy Hall',      'Investment Banking', 'MD',                '2013-02-01', 'Active'),
(19, 'Mark Young',      'Risk Management',    'Risk Manager',      '2017-10-10', 'Active'),
(20, 'Lisa Allen',      'Compliance',         'VP',                '2015-06-01', 'Active');

-- --------------------------------------------
-- BROKERDEALER
-- --------------------------------------------
INSERT INTO BrokerDealer (BrokerDealerID, BrokerDealerName, Country, RegistrationNumber) VALUES
(1, 'Fidelity',       'USA', 'FD123'),
(2, 'Charles Schwab', 'USA', 'CS456'),
(3, 'Morgan Stanley', 'USA', 'MS789'),
(4, 'ETrade',         'USA', 'ET111'),
(5, 'Robinhood',      'USA', 'RH222');

-- --------------------------------------------
-- ACCOUNT
-- --------------------------------------------
INSERT INTO Account (AccountID, EmployeeID, BrokerDealerID, AccountNumber, AccountType, OpenDate, Status) VALUES
(1,  1,  1, 'ACC1',  'Individual', '2022-01-01', 'Active'),
(2,  2,  2, 'ACC2',  'Individual', '2022-01-01', 'Active'),
(3,  3,  3, 'ACC3',  'Individual', '2022-01-01', 'Active'),
(4,  4,  4, 'ACC4',  'Individual', '2022-01-01', 'Active'),
(5,  5,  5, 'ACC5',  'Individual', '2022-01-01', 'Active'),
(6,  6,  1, 'ACC6',  'Individual', '2022-01-01', 'Active'),
(7,  7,  2, 'ACC7',  'Individual', '2022-01-01', 'Active'),
(8,  8,  3, 'ACC8',  'Individual', '2022-01-01', 'Active'),
(9,  9,  4, 'ACC9',  'Individual', '2022-01-01', 'Active'),
(10, 10, 5, 'ACC10', 'Individual', '2022-01-01', 'Active'),
(11, 11, 1, 'ACC11', 'Individual', '2022-01-01', 'Active'),
(12, 12, 2, 'ACC12', 'Individual', '2022-01-01', 'Active'),
(13, 13, 3, 'ACC13', 'Individual', '2022-01-01', 'Active'),
(14, 14, 4, 'ACC14', 'Individual', '2022-01-01', 'Active'),
(15, 15, 5, 'ACC15', 'Individual', '2022-01-01', 'Active'),
(16, 16, 1, 'ACC16', 'Individual', '2022-01-01', 'Active'),
(17, 17, 2, 'ACC17', 'Individual', '2022-01-01', 'Active'),
(18, 18, 3, 'ACC18', 'Individual', '2022-01-01', 'Active'),
(19, 19, 4, 'ACC19', 'Individual', '2022-01-01', 'Active'),
(20, 20, 5, 'ACC20', 'Individual', '2022-01-01', 'Active');

-- --------------------------------------------
-- RESTRICTEDSECURITY
-- --------------------------------------------
INSERT INTO RestrictedSecurity (RestrictionID, SecuritySymbol, RestrictionType, StartDate, EndDate, Reason, AddedBy) VALUES
(1,  'AAPL', 'Insider List', '2025-10-01', NULL,         'Pending earnings announcement',    'Linda Thomas'),
(2,  'MSFT', 'Blackout',     '2025-11-01', '2025-11-30', 'Quarterly blackout period',         'Linda Thomas'),
(3,  'GOOG', 'Watch List',   '2025-09-15', NULL,         'Elevated insider activity detected','Mark Young'),
(4,  'TSLA', 'Insider List', '2026-01-01', NULL,         'Board member transaction window',   'Lisa Allen'),
(5,  'AMZN', 'Blackout',     '2025-12-01', '2025-12-31', 'Year-end blackout',                 'Linda Thomas'),
(6,  'META', 'Watch List',   '2026-02-01', NULL,         'Regulatory review underway',        'Mark Young'),
(7,  'NVDA', 'Insider List', '2026-01-15', NULL,         'Pre-earnings window',               'Lisa Allen'),
(8,  'NFLX', 'Blackout',     '2025-10-15', '2025-10-31', 'Content deal announcement',         'Linda Thomas');

-- --------------------------------------------
-- TRADEREQUEST
-- --------------------------------------------
INSERT INTO TradeRequest (TradeRequestID, EmployeeID, BrokerDealerID, RequestDate, SecuritySymbol, TradeType, Quantity, Status) VALUES
(1001, 1,  1, '2025-10-05', 'IBM',  'BUY',  100, 'Approved'),
(1002, 2,  2, '2025-10-06', 'IBM',  'SELL',  50, 'Approved'),
(1003, 3,  3, '2025-10-07', 'AMZN', 'BUY',   75, 'Approved'),
(1004, 4,  4, '2025-10-08', 'TSLA', 'BUY',  200, 'Approved'),
(1005, 5,  5, '2025-10-09', 'NFLX', 'SELL', 150, 'Approved'),
(1006, 1,  1, '2025-10-10', 'AAPL', 'BUY',  500, 'Rejected'),
(1007, 7,  2, '2025-11-05', 'MSFT', 'BUY',  300, 'Rejected'),
(1008, 4,  4, '2026-01-10', 'TSLA', 'SELL', 400, 'Escalated'),
(1009, 5,  5, '2025-10-20', 'NFLX', 'BUY',  250, 'Rejected'),
(1010, 11, 1, '2026-02-10', 'META', 'BUY',  600, 'Escalated'),
(1011, 8,  3, '2025-11-15', 'GOOG', 'BUY',  2000, 'Approved'),
(1012, 8,  3, '2025-11-16', 'GOOG', 'BUY',  1800, 'Approved'),
(1013, 8,  3, '2025-11-17', 'GOOG', 'BUY',  2200, 'Escalated'),
(1014, 5,  5, '2026-01-10', 'IBM',  'BUY',   80, 'Approved'),
(1015, 6,  1, '2026-01-15', 'IBM',  'SELL',  60, 'Approved'),
(1016, 7,  2, '2026-02-05', 'GOOG', 'BUY',   85, 'Approved'),
(1017, 8,  3, '2026-02-05', 'GOOG', 'SELL', 100, 'Approved'),
(1018, 9,  4, '2026-02-05', 'GOOG', 'BUY',  115, 'Approved'),
(1019, 10, 5, '2026-02-05', 'GOOG', 'BUY',  130, 'Approved'),
(1020, 12, 2, '2026-02-20', 'NVDA', 'BUY',  700, 'Rejected'),
(1021, 15, 5, '2026-03-01', 'IBM',  'BUY',   90, 'Approved'),
(1022, 17, 2, '2026-03-05', 'IBM',  'SELL',  45, 'Pending'),
(1023, 18, 3, '2026-03-10', 'TSLA', 'BUY',  300, 'Escalated'),
(1024, 4,  4, '2026-03-12', 'AAPL', 'SELL', 150, 'Rejected'),
(1025, 11, 1, '2026-03-15', 'META', 'SELL', 500, 'Escalated');

-- --------------------------------------------
-- COMPLIANCEALERT
-- --------------------------------------------
INSERT INTO ComplianceAlert (AlertID, TradeRequestID, EmployeeID, AlertType, AlertDate, Severity, Status, Description, ResolvedDate) VALUES
(1,  1006, 1,  'Restricted Security', '2025-10-10', 'High',     'Closed',        'Employee traded AAPL during insider list window',          '2025-10-15'),
(2,  1007, 7,  'Restricted Security', '2025-11-05', 'High',     'Closed',        'MSFT trade attempted during blackout period',               '2025-11-10'),
(3,  1008, 4,  'Insider Trading',     '2026-01-10', 'Critical', 'Investigating', 'Large TSLA sell during insider restriction - under review',  NULL),
(4,  1009, 5,  'Restricted Security', '2025-10-20', 'Medium',   'Closed',        'NFLX trade during content deal blackout',                   '2025-10-25'),
(5,  1010, 11, 'Insider Trading',     '2026-02-10', 'Critical', 'Investigating', 'META buy during regulatory review - escalated to legal',     NULL),
(6,  1013, 8,  'Excessive Volume',    '2025-11-17', 'High',     'Closed',        'GOOG volume exceeded 2000 units threshold 3 days running',  '2025-11-25'),
(7,  1020, 12, 'Restricted Security', '2026-02-20', 'High',     'Open',          'NVDA trade attempted during pre-earnings insider window',   NULL),
(8,  1023, 18, 'Insider Trading',     '2026-03-10', 'Critical', 'Investigating', 'Senior MD submitted TSLA buy during active restriction',    NULL),
(9,  1024, 4,  'Restricted Security', '2026-03-12', 'High',     'Open',          'Repeat AAPL violation - same employee as alert #1',         NULL),
(10, 1025, 11, 'Unusual Pattern',     '2026-03-15', 'Medium',   'Open',          'Second META escalation for this employee in 30 days',       NULL),
(11, NULL,  5,  'Unusual Pattern',    '2025-12-01', 'Medium',   'Closed',        'Unusual login pattern during off-hours - investigated',     '2025-12-05'),
(12, NULL,  8,  'Excessive Volume',   '2025-12-10', 'Low',      'Closed',        'Monthly volume limit advisory issued',                      '2025-12-12'),
(13, NULL,  1,  'Unusual Pattern',    '2026-01-20', 'Medium',   'Investigating', 'Multiple requests across same security within short window', NULL),
(14, NULL,  4,  'Insider Trading',    '2026-02-01', 'High',     'Investigating', 'Third-party tip received regarding employee trading pattern', NULL),
(15, NULL, 18,  'Unusual Pattern',    '2026-03-01', 'Low',      'Open',          'MD account linked to two brokers with concurrent activity',  NULL);

-- --------------------------------------------
-- APPROVALWORKFLOW
-- --------------------------------------------
INSERT INTO ApprovalWorkflow (WorkflowID, TradeRequestID, ReviewerID, ReviewDate, Decision, Comments, TurnaroundDays) VALUES
(1,  1001, 2,  '2025-10-06', 'Approved',  'Standard request, no issues',                        1),
(2,  1002, 6,  '2025-10-07', 'Approved',  'Verified employee clearance',                        1),
(3,  1003, 10, '2025-10-08', 'Approved',  'Within policy limits',                               1),
(4,  1004, 2,  '2025-10-09', 'Approved',  'Normal trade volume',                                1),
(5,  1005, 16, '2025-10-10', 'Approved',  'No restrictions found',                              1),
(6,  1006, 10, '2025-10-10', 'Rejected',  'AAPL is on insider list - trade denied',             0),
(7,  1007, 20, '2025-11-05', 'Rejected',  'MSFT blackout period active until Nov 30',           0),
(8,  1008, 19, '2026-01-12', 'Escalated', 'Escalated to Legal - TSLA insider restriction',      2),
(9,  1009, 6,  '2025-10-21', 'Rejected',  'NFLX in blackout - content deal window',             1),
(10, 1010, 10, '2026-02-12', 'Escalated', 'Second META violation - escalated to Compliance VP', 2),
(11, 1011, 2,  '2025-11-16', 'Approved',  'High volume but within limits',                      1),
(12, 1012, 2,  '2025-11-17', 'Approved',  'Volume elevated, approved with note',                1),
(13, 1013, 10, '2025-11-19', 'Escalated', 'Third consecutive high-volume GOOG - escalated',     2),
(14, 1014, 16, '2026-01-11', 'Approved',  'Routine',                                            1),
(15, 1015, 6,  '2026-01-16', 'Approved',  'Verified, within limits',                            1),
(16, 1016, 2,  '2026-02-06', 'Approved',  'No issues',                                          1),
(17, 1017, 9,  '2026-02-07', 'Approved',  'Normal sell',                                        2),
(18, 1018, 14, '2026-02-08', 'Approved',  'Within risk tolerance',                              3),
(19, 1019, 20, '2026-02-06', 'Approved',  'Cleared',                                            1),
(20, 1020, 10, '2026-02-21', 'Rejected',  'NVDA on insider list - pre-earnings window',         1),
(21, 1021, 6,  '2026-03-02', 'Approved',  'Standard',                                           1),
(22, 1022, 2,  NULL,         'Pending',   'Awaiting senior review',                             NULL),
(23, 1023, 19, '2026-03-12', 'Escalated', 'MD-level TSLA buy escalated immediately to Director',2),
(24, 1024, 16, '2026-03-13', 'Rejected',  'Repeat AAPL violation - flagged for pattern review', 1),
(25, 1025, 20, '2026-03-16', 'Escalated', 'Repeat META pattern - Compliance VP review',         1);


-- =============================================
-- ADDITIONAL DATA
-- =============================================

INSERT INTO BrokerDealer (BrokerDealerID, BrokerDealerName, Country, RegistrationNumber) VALUES
(6,  'Goldman Sachs',       'USA', 'GS333'),
(7,  'TD Ameritrade',       'USA', 'TD444'),
(8,  'Interactive Brokers', 'USA', 'IB555'),
(9,  'Barclays',            'UK',  'BC666'),
(10, 'Deutsche Bank',       'DE',  'DB777');

INSERT INTO Employee (EmployeeID, EmployeeName, Department, JobTitle, HireDate, Status) VALUES
(21, 'Carlos Rivera',   'Legal',              'Counsel',            '2020-08-01', 'Active'),
(22, 'Megan Foster',    'Operations',         'Operations Analyst', '2021-11-15', 'Active'),
(23, 'George Kim',      'Trading',            'Senior Trader',      '2016-05-01', 'Active'),
(24, 'Rachel Patel',    'Investment Banking', 'Associate',          '2023-01-10', 'Active'),
(25, 'Tom Nguyen',      'Risk Management',    'Risk Analyst',       '2022-04-01', 'Active'),
(26, 'Diane Scott',     'Compliance',         'Senior Analyst',     '2019-07-20', 'Active'),
(27, 'Alex Turner',     'Technology',         'Lead Engineer',      '2017-03-15', 'Active'),
(28, 'Samantha Green',  'Investment Banking', 'Analyst',            '2024-02-01', 'Active'),
(29, 'Brian Cooper',    'Trading',            'Trader',             '2023-09-01', 'Active'),
(30, 'Michelle Reed',   'Compliance',         'Analyst',            '2021-05-10', 'Terminated');

INSERT INTO Account (AccountID, EmployeeID, BrokerDealerID, AccountNumber, AccountType, OpenDate, Status) VALUES
(21, 23, 6,  'ACC21', 'Individual', '2021-03-01', 'Active'),
(22, 24, 7,  'ACC22', 'Individual', '2023-02-01', 'Active'),
(23, 25, 8,  'ACC23', 'Individual', '2022-05-01', 'Active'),
(24, 26, 9,  'ACC24', 'Individual', '2019-09-01', 'Active'),
(25, 27, 10, 'ACC25', 'Individual', '2017-05-01', 'Active'),
(26, 28, 6,  'ACC26', 'Individual', '2024-03-01', 'Active'),
(27, 29, 7,  'ACC27', 'Individual', '2023-10-01', 'Active'),
(28, 1,  6,  'ACC28', 'Retirement', '2023-06-01', 'Active'),
(29, 8,  6,  'ACC29', 'Joint',      '2022-08-01', 'Active'),
(30, 11, 8,  'ACC30', 'Individual', '2023-05-01', 'Active');

INSERT INTO TradeRequest (TradeRequestID, EmployeeID, BrokerDealerID, RequestDate, SecuritySymbol, TradeType, Quantity, Status) VALUES
(1026, 1,  1,  '2024-01-10', 'IBM',  'BUY',   120, 'Approved'),
(1027, 5,  5,  '2024-01-20', 'AMZN', 'BUY',   200, 'Approved'),
(1028, 8,  3,  '2024-02-05', 'GOOG', 'BUY',   600, 'Approved'),
(1029, 7,  2,  '2024-02-15', 'MSFT', 'BUY',   150, 'Approved'),
(1030, 4,  4,  '2024-03-01', 'TSLA', 'BUY',   250, 'Approved'),
(1031, 12, 2,  '2024-03-10', 'AAPL', 'BUY',   100, 'Approved'),
(1032, 18, 3,  '2024-04-01', 'GOOG', 'BUY',   400, 'Approved'),
(1033, 8,  3,  '2024-04-15', 'GOOG', 'BUY',   750, 'Approved'),
(1034, 15, 5,  '2024-05-01', 'NVDA', 'BUY',   200, 'Approved'),
(1035, 1,  6,  '2024-05-15', 'IBM',  'BUY',   110, 'Approved'),
(1036, 23, 6,  '2024-06-01', 'AMZN', 'BUY',    85, 'Approved'),
(1037, 24, 7,  '2024-06-15', 'IBM',  'BUY',    70, 'Approved'),
(1038, 8,  3,  '2024-07-01', 'GOOG', 'BUY',   900, 'Escalated'),
(1039, 25, 8,  '2024-07-10', 'GOOG', 'BUY',   120, 'Approved'),
(1040, 11, 1,  '2024-07-20', 'META', 'BUY',   300, 'Approved'),
(1041, 5,  5,  '2024-08-01', 'TSLA', 'SELL',  150, 'Approved'),
(1042, 26, 9,  '2024-08-15', 'IBM',  'BUY',    90, 'Approved'),
(1043, 27, 10, '2024-09-01', 'GOOG', 'SELL',  130, 'Approved'),
(1044, 29, 7,  '2024-09-15', 'IBM',  'BUY',    65, 'Approved'),
(1045, 28, 6,  '2024-10-01', 'AMZN', 'SELL',   95, 'Approved'),
(1046, 1,  1,  '2025-01-05', 'IBM',  'BUY',   130, 'Approved'),
(1047, 5,  5,  '2025-01-20', 'IBM',  'SELL',   80, 'Approved'),
(1048, 23, 6,  '2025-02-01', 'GOOG', 'BUY',    95, 'Approved'),
(1049, 24, 7,  '2025-03-01', 'AMZN', 'BUY',    70, 'Approved'),
(1050, 26, 9,  '2025-03-15', 'IBM',  'BUY',   100, 'Approved'),
(1051, 8,  3,  '2025-04-01', 'GOOG', 'BUY',  1100, 'Escalated'),
(1052, 27, 10, '2025-04-15', 'GOOG', 'SELL',  130, 'Approved'),
(1053, 29, 7,  '2025-05-01', 'IBM',  'BUY',    75, 'Approved'),
(1054, 4,  4,  '2025-05-15', 'TSLA', 'BUY',   180, 'Approved'),
(1055, 7,  2,  '2025-06-01', 'GOOG', 'BUY',    60, 'Approved'),
(1056, 12, 2,  '2025-06-15', 'NVDA', 'BUY',   400, 'Approved'),
(1057, 15, 5,  '2025-07-01', 'IBM',  'SELL',   55, 'Approved'),
(1058, 18, 3,  '2025-07-15', 'AMZN', 'BUY',   110, 'Approved'),
(1059, 11, 1,  '2025-08-01', 'META', 'SELL',  250, 'Approved'),
(1060, 5,  5,  '2025-09-01', 'IBM',  'BUY',    95, 'Approved'),
(1061, 28, 6,  '2025-09-10', 'AMZN', 'BUY',    80, 'Approved'),
(1062, 25, 8,  '2025-09-20', 'IBM',  'BUY',    70, 'Approved'),
(1063, 4,  4,  '2025-10-01', 'IBM',  'BUY',   160, 'Approved'),
(1064, 29, 7,  '2025-10-15', 'IBM',  'SELL',   55, 'Approved'),
(1065, 23, 6,  '2025-10-20', 'AMZN', 'BUY',    75, 'Approved'),
(1066, 23, 6,  '2026-01-05', 'IBM',  'BUY',    85, 'Approved'),
(1067, 26, 9,  '2026-01-20', 'IBM',  'BUY',    75, 'Approved'),
(1068, 4,  4,  '2026-03-01', 'AAPL', 'BUY',   250, 'Rejected'),
(1069, 7,  2,  '2026-03-05', 'NVDA', 'SELL',  350, 'Rejected'),
(1070, 8,  3,  '2026-03-15', 'GOOG', 'BUY',  3000, 'Escalated'),
(1071, 15, 5,  '2026-03-20', 'META', 'BUY',   400, 'Rejected'),
(1072, 29, 7,  '2026-03-25', 'IBM',  'BUY',    65, 'Pending'),
(1073, 25, 8,  '2026-03-25', 'IBM',  'BUY',    70, 'Approved'),
(1074, 1,  6,  '2026-04-01', 'IBM',  'BUY',   140, 'Pending'),
(1075, 5,  5,  '2026-04-05', 'IBM',  'SELL',  110, 'Pending');

INSERT INTO ComplianceAlert (AlertID, TradeRequestID, EmployeeID, AlertType, AlertDate, Severity, Status, Description, ResolvedDate) VALUES
(16, 1028, 8,  'Excessive Volume',    '2024-02-05', 'Medium',   'Closed',        'GOOG volume advisory — 600 units flagged for monitoring',                   '2024-02-10'),
(17, 1033, 8,  'Excessive Volume',    '2024-04-15', 'High',     'Closed',        'GOOG volume escalating: 750 units, second alert in 10 weeks',               '2024-04-22'),
(18, 1038, 8,  'Excessive Volume',    '2024-07-01', 'High',     'Closed',        'Third high-volume GOOG batch — 900 units — escalated to compliance review', '2024-07-15'),
(19, 1051, 8,  'Excessive Volume',    '2025-04-01', 'High',     'Closed',        'Volume pattern resumes in 2025 — 1100 GOOG units, fourth escalation',       '2025-04-14'),
(20, 1068, 4,  'Restricted Security', '2026-03-01', 'High',     'Open',          'Third AAPL attempt for same employee — pattern review initiated',           NULL),
(21, 1069, 7,  'Restricted Security', '2026-03-05', 'High',     'Investigating', 'NVDA trade attempted during pre-earnings insider restriction window',        NULL),
(22, 1070, 8,  'Excessive Volume',    '2026-03-15', 'Critical', 'Investigating', '3000-unit GOOG order — unprecedented single-day volume, referred to Legal',  NULL),
(23, 1071, 15, 'Insider Trading',     '2026-03-20', 'High',     'Open',          'META trade during regulatory review watch-list period — second offense',     NULL),
(24, NULL,  4,  'Insider Trading',    '2026-03-08', 'Critical', 'Investigating', 'Pattern review: Emily Davis — three restricted-security attempts in 6 months', NULL),
(25, NULL,  8,  'Excessive Volume',   '2024-08-20', 'Low',      'Closed',        'Monthly volume summary advisory issued after Q2-Q3 GOOG activity',           '2024-08-25'),
(26, NULL,  23, 'Unusual Pattern',    '2025-10-22', 'Low',      'Closed',        'Same-day trades detected at two brokers — George Kim — investigated',         '2025-10-27'),
(27, NULL,  26, 'Unusual Pattern',    '2025-03-16', 'Medium',   'Closed',        'Cross-border Barclays account activity flagged during onboarding review',     '2025-03-21'),
(28, NULL,  11, 'Unusual Pattern',    '2025-08-02', 'Low',      'Closed',        'Concurrent sell/buy on META within 48-hour window — no policy breach found',  '2025-08-07'),
(29, NULL,  27, 'Unusual Pattern',    '2026-02-12', 'Medium',   'Open',          'Multiple broker activity across Deutsche Bank and in-house account — 48h window', NULL),
(30, NULL,  29, 'Unusual Pattern',    '2026-03-26', 'Low',      'Open',          'New employee account — three trades in one week — monitoring initiated',      NULL),
(31, NULL,  5,  'Unusual Pattern',    '2026-04-05', 'Low',      'Open',          'Pending IBM sell overlaps active buy position — flagged for review',          NULL),
(32, NULL,  1,  'Unusual Pattern',    '2026-04-01', 'Low',      'Open',          'Second IBM purchase via Goldman Sachs retirement account within 60 days',     NULL),
(33, NULL,  12, 'Restricted Security','2026-02-21', 'Medium',   'Open',          'Follow-up advisory: NVDA restriction window now active — account monitored',  NULL),
(34, NULL,  18, 'Insider Trading',    '2026-03-15', 'Medium',   'Investigating', 'Third-party tip received regarding Nancy Hall trading activity — referred to Legal', NULL),
(35, NULL,  15, 'Insider Trading',    '2026-03-22', 'Medium',   'Open',          'Second META-related alert for Daniel Lewis within 30 days',                  NULL);

INSERT INTO ApprovalWorkflow (WorkflowID, TradeRequestID, ReviewerID, ReviewDate, Decision, Comments, TurnaroundDays) VALUES
(26, 1026, 6,  '2024-01-11', 'Approved',  'Standard IBM trade — no issues',                                           1),
(27, 1028, 2,  '2024-02-08', 'Approved',  'High volume noted — within 2024 threshold, advisory issued',               3),
(28, 1033, 10, '2024-04-18', 'Approved',  'Volume elevated, approved with written warning',                           3),
(29, 1038, 10, '2024-07-08', 'Escalated', 'GOOG volume exceeds cumulative limit — escalated to Director',             7),
(30, 1040, 6,  '2024-07-23', 'Approved',  'META trade cleared — no restriction in place',                             3),
(31, 1041, 16, '2024-08-04', 'Approved',  'Normal sell within approved limits',                                       3),
(32, 1042, 26, '2024-08-22', 'Approved',  'First Barclays account review — cleared after extended due diligence',     7),
(33, 1044, 9,  '2024-09-19', 'Approved',  'Brian Cooper new account — additional KYC check completed',                4),
(34, 1046, 6,  '2025-01-09', 'Approved',  'Routine IBM buy — cleared',                                                4),
(35, 1047, 16, '2025-01-24', 'Approved',  'Within sell limits — approved',                                            4),
(36, 1051, 10, '2025-04-10', 'Escalated', 'Volume pattern across 4 months — escalated to Director and Legal review',  9),
(37, 1053, 2,  '2025-05-05', 'Approved',  'Standard Brian Cooper review',                                             4),
(38, 1054, 9,  '2025-05-22', 'Approved',  'TSLA pre-restriction — cleared with note that restriction starts Jan 2026', 7),
(39, 1056, 19, '2025-06-26', 'Approved',  'NVDA not yet restricted — approved; restriction window noted for Jan 2026', 11),
(40, 1057, 6,  '2025-07-05', 'Approved',  'Routine IBM sell',                                                         4),
(41, 1058, 14, '2025-07-22', 'Approved',  'AMZN within policy — pre-blackout period',                                 7),
(42, 1060, 2,  '2025-09-05', 'Approved',  'Cleared',                                                                  4),
(43, 1063, 26, '2025-10-06', 'Approved',  'Normal volume, no restriction conflicts',                                   5),
(44, 1064, 6,  '2025-10-18', 'Approved',  'IBM sell — within limits',                                                 3),
(45, 1065, 16, '2025-10-23', 'Approved',  'AMZN buy approved — blackout window does not start until Dec 1',           3),
(46, 1066, 2,  '2026-01-08', 'Approved',  'Standard Goldman Sachs account review',                                    3),
(47, 1067, 26, '2026-01-23', 'Approved',  'Barclays IBM trade — no restriction conflicts',                             3),
(48, 1068, 10, '2026-03-06', 'Rejected',  'AAPL insider list active — third violation for this employee, referred to Legal', 5),
(49, 1069, 19, '2026-03-10', 'Rejected',  'NVDA pre-earnings restriction in force — trade denied, employee notified',  5),
(50, 1070, 10, '2026-03-29', 'Escalated', 'Extreme GOOG volume 3000 units — Director, Legal, and Risk reviewed; decision pending', 14),
(51, 1071, 20, '2026-03-27', 'Rejected',  'META watch-list restriction — flagged employee, second offense escalated',  7),
(52, 1073, 9,  '2026-03-29', 'Approved',  'IBM within limits — Interactive Brokers account cleared',                  4),
(53, 1035, 2,  '2024-05-18', 'Approved',  'Goldman Sachs account first review — cleared',                             3),
(54, 1036, 6,  '2024-06-04', 'Approved',  'George Kim new employee account — no issues',                              3),
(55, 1037, 16, '2024-06-18', 'Approved',  'Rachel Patel — TD Ameritrade account approved after standard review',      3)
