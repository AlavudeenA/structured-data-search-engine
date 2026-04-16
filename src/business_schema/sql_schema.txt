-- =============================================
-- COMPLIANCE DATABASE - FULL DDL + SAMPLE DATA
-- Tables: Employee, BrokerDealer, Account,
--         TradeRequest, ComplianceAlert,
--         RestrictedSecurity, ApprovalWorkflow
-- =============================================

USE [Compliance]
GO

-- =============================================
-- DROP existing tables (safe order)
-- =============================================
IF OBJECT_ID('dbo.ApprovalWorkflow',  'U') IS NOT NULL DROP TABLE dbo.ApprovalWorkflow;
IF OBJECT_ID('dbo.ComplianceAlert',   'U') IS NOT NULL DROP TABLE dbo.ComplianceAlert;
IF OBJECT_ID('dbo.TradeRequest',      'U') IS NOT NULL DROP TABLE dbo.TradeRequest;
IF OBJECT_ID('dbo.RestrictedSecurity','U') IS NOT NULL DROP TABLE dbo.RestrictedSecurity;
IF OBJECT_ID('dbo.Account',           'U') IS NOT NULL DROP TABLE dbo.Account;
IF OBJECT_ID('dbo.BrokerDealer',      'U') IS NOT NULL DROP TABLE dbo.BrokerDealer;
IF OBJECT_ID('dbo.Employee',          'U') IS NOT NULL DROP TABLE dbo.Employee;
GO

-- =============================================
-- 1. EMPLOYEE
-- =============================================
CREATE TABLE dbo.Employee (
    EmployeeID   INT          NOT NULL PRIMARY KEY,
    EmployeeName VARCHAR(100) NOT NULL,
    Department   VARCHAR(100) NULL,
    JobTitle     VARCHAR(100) NULL,
    HireDate     DATE         NULL,
    Status       VARCHAR(20)  NULL   -- Active, Terminated
);
GO

-- =============================================
-- 2. BROKERDEALER
-- =============================================
CREATE TABLE dbo.BrokerDealer (
    BrokerDealerID   INT          NOT NULL PRIMARY KEY,
    BrokerDealerName VARCHAR(200) NOT NULL,
    Country          VARCHAR(100) NULL,
    RegistrationNumber VARCHAR(100) NULL
);
GO

-- =============================================
-- 3. ACCOUNT
-- =============================================
CREATE TABLE dbo.Account (
    AccountID      INT         NOT NULL PRIMARY KEY,
    EmployeeID     INT         NOT NULL REFERENCES dbo.Employee(EmployeeID),
    BrokerDealerID INT         NOT NULL REFERENCES dbo.BrokerDealer(BrokerDealerID),
    AccountNumber  VARCHAR(50) NULL,
    AccountType    VARCHAR(50) NULL,
    OpenDate       DATE        NULL,
    Status         VARCHAR(20) NULL   -- Active, Closed
);
GO

-- =============================================
-- 4. RESTRICTEDSECURITY
-- =============================================
CREATE TABLE dbo.RestrictedSecurity (
    RestrictionID    INT          NOT NULL PRIMARY KEY,
    SecuritySymbol   VARCHAR(20)  NOT NULL,
    RestrictionType  VARCHAR(100) NULL,   -- Blackout, Insider List, Watch List
    StartDate        DATE         NULL,
    EndDate          DATE         NULL,   -- NULL = still active
    Reason           VARCHAR(200) NULL,
    AddedBy          VARCHAR(100) NULL
);
GO

-- =============================================
-- 5. TRADEREQUEST  (BrokerDealerID added directly)
-- =============================================
CREATE TABLE dbo.TradeRequest (
    TradeRequestID INT         NOT NULL PRIMARY KEY,
    EmployeeID     INT         NOT NULL REFERENCES dbo.Employee(EmployeeID),
    BrokerDealerID INT         NOT NULL REFERENCES dbo.BrokerDealer(BrokerDealerID),
    RequestDate    DATE        NULL,
    SecuritySymbol VARCHAR(20) NULL,
    TradeType      VARCHAR(10) NULL,   -- BUY, SELL
    Quantity       INT         NULL,
    Status         VARCHAR(20) NULL    -- Pending, Approved, Rejected, Escalated
);
GO

-- =============================================
-- 6. COMPLIANCEALERT
-- =============================================
CREATE TABLE dbo.ComplianceAlert (
    AlertID        INT          NOT NULL PRIMARY KEY,
    TradeRequestID INT          NULL REFERENCES dbo.TradeRequest(TradeRequestID),
    EmployeeID     INT          NOT NULL REFERENCES dbo.Employee(EmployeeID),
    AlertType      VARCHAR(100) NULL,   -- Insider Trading, Excessive Volume,
                                        -- Restricted Security, Unusual Pattern
    AlertDate      DATE         NULL,
    Severity       VARCHAR(20)  NULL,   -- Low, Medium, High, Critical
    Status         VARCHAR(20)  NULL,   -- Open, Investigating, Closed, Escalated
    Description    VARCHAR(500) NULL,
    ResolvedDate   DATE         NULL
);
GO

-- =============================================
-- 7. APPROVALWORKFLOW
-- =============================================
CREATE TABLE dbo.ApprovalWorkflow (
    WorkflowID     INT          NOT NULL PRIMARY KEY,
    TradeRequestID INT          NOT NULL REFERENCES dbo.TradeRequest(TradeRequestID),
    ReviewerID     INT          NOT NULL REFERENCES dbo.Employee(EmployeeID),
    ReviewDate     DATE         NULL,
    Decision       VARCHAR(20)  NULL,   -- Approved, Rejected, Escalated, Pending
    Comments       VARCHAR(500) NULL,
    TurnaroundDays INT          NULL    -- ReviewDate - RequestDate
);
GO


-- =============================================
-- SAMPLE DATA
-- =============================================

-- --------------------------------------------
-- EMPLOYEE  (20 employees, mixed departments)
-- --------------------------------------------
INSERT INTO dbo.Employee VALUES
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
GO

-- --------------------------------------------
-- BROKERDEALER
-- --------------------------------------------
INSERT INTO dbo.BrokerDealer VALUES
(1, 'Fidelity',       'USA', 'FD123'),
(2, 'Charles Schwab', 'USA', 'CS456'),
(3, 'Morgan Stanley', 'USA', 'MS789'),
(4, 'ETrade',         'USA', 'ET111'),
(5, 'Robinhood',      'USA', 'RH222');
GO

-- --------------------------------------------
-- ACCOUNT  (employees linked to broker dealers)
-- --------------------------------------------
INSERT INTO dbo.Account VALUES
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
GO

-- --------------------------------------------
-- RESTRICTEDSECURITY
-- (some active, some expired — good for capsule testing)
-- --------------------------------------------
INSERT INTO dbo.RestrictedSecurity VALUES
(1,  'AAPL', 'Insider List', '2025-10-01', NULL,         'Pending earnings announcement',    'Linda Thomas'),
(2,  'MSFT', 'Blackout',     '2025-11-01', '2025-11-30', 'Quarterly blackout period',         'Linda Thomas'),
(3,  'GOOG', 'Watch List',   '2025-09-15', NULL,         'Elevated insider activity detected','Mark Young'),
(4,  'TSLA', 'Insider List', '2026-01-01', NULL,         'Board member transaction window',   'Lisa Allen'),
(5,  'AMZN', 'Blackout',     '2025-12-01', '2025-12-31', 'Year-end blackout',                 'Linda Thomas'),
(6,  'META', 'Watch List',   '2026-02-01', NULL,         'Regulatory review underway',        'Mark Young'),
(7,  'NVDA', 'Insider List', '2026-01-15', NULL,         'Pre-earnings window',               'Lisa Allen'),
(8,  'NFLX', 'Blackout',     '2025-10-15', '2025-10-31', 'Content deal announcement',         'Linda Thomas');
GO

-- --------------------------------------------
-- TRADEREQUEST
-- (deliberate policy violations for alert testing)
-- --------------------------------------------
INSERT INTO dbo.TradeRequest VALUES
-- Normal approved requests
(1001, 1,  1, '2025-10-05', 'IBM',  'BUY',  100, 'Approved'),
(1002, 2,  2, '2025-10-06', 'IBM',  'SELL',  50, 'Approved'),
(1003, 3,  3, '2025-10-07', 'AMZN', 'BUY',   75, 'Approved'),  -- AMZN not yet restricted
(1004, 4,  4, '2025-10-08', 'TSLA', 'BUY',  200, 'Approved'),  -- TSLA not yet restricted
(1005, 5,  5, '2025-10-09', 'NFLX', 'SELL', 150, 'Approved'),
-- Requests on restricted securities (violations)
(1006, 1,  1, '2025-10-10', 'AAPL', 'BUY',  500, 'Rejected'),  -- AAPL on Insider List
(1007, 7,  2, '2025-11-05', 'MSFT', 'BUY',  300, 'Rejected'),  -- MSFT in Blackout
(1008, 4,  4, '2026-01-10', 'TSLA', 'SELL', 400, 'Escalated'), -- TSLA Insider List
(1009, 5,  5, '2025-10-20', 'NFLX', 'BUY',  250, 'Rejected'),  -- NFLX in Blackout
(1010, 11, 1, '2026-02-10', 'META', 'BUY',  600, 'Escalated'), -- META Watch List
-- Excessive volume requests
(1011, 8,  3, '2025-11-15', 'GOOG', 'BUY',  2000, 'Approved'),
(1012, 8,  3, '2025-11-16', 'GOOG', 'BUY',  1800, 'Approved'),
(1013, 8,  3, '2025-11-17', 'GOOG', 'BUY',  2200, 'Escalated'),
-- Recent requests (2026 - for trend capsules)
(1014, 5,  5, '2026-01-10', 'IBM',  'BUY',   80, 'Approved'),
(1015, 6,  1, '2026-01-15', 'IBM',  'SELL',  60, 'Approved'),
(1016, 7,  2, '2026-02-05', 'GOOG', 'BUY',   85, 'Approved'),
(1017, 8,  3, '2026-02-05', 'GOOG', 'SELL', 100, 'Approved'),
(1018, 9,  4, '2026-02-05', 'GOOG', 'BUY',  115, 'Approved'),
(1019, 10, 5, '2026-02-05', 'GOOG', 'BUY',  130, 'Approved'),
(1020, 12, 2, '2026-02-20', 'NVDA', 'BUY',  700, 'Rejected'),  -- NVDA Insider List
(1021, 15, 5, '2026-03-01', 'IBM',  'BUY',   90, 'Approved'),
(1022, 17, 2, '2026-03-05', 'IBM',  'SELL',  45, 'Pending'),
(1023, 18, 3, '2026-03-10', 'TSLA', 'BUY',  300, 'Escalated'), -- TSLA Insider List
(1024, 4,  4, '2026-03-12', 'AAPL', 'SELL', 150, 'Rejected'),  -- AAPL Insider List
(1025, 11, 1, '2026-03-15', 'META', 'SELL', 500, 'Escalated'); -- META Watch List
GO

-- --------------------------------------------
-- COMPLIANCEALERT
-- (tied to violations above + some standalone)
-- --------------------------------------------
INSERT INTO dbo.ComplianceAlert VALUES
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
-- Standalone alerts (no trade request linked)
(11, NULL,  5,  'Unusual Pattern',    '2025-12-01', 'Medium',   'Closed',        'Unusual login pattern during off-hours - investigated',     '2025-12-05'),
(12, NULL,  8,  'Excessive Volume',   '2025-12-10', 'Low',      'Closed',        'Monthly volume limit advisory issued',                      '2025-12-12'),
(13, NULL,  1,  'Unusual Pattern',    '2026-01-20', 'Medium',   'Investigating', 'Multiple requests across same security within short window', NULL),
(14, NULL,  4,  'Insider Trading',    '2026-02-01', 'High',     'Investigating', 'Third-party tip received regarding employee trading pattern', NULL),
(15, NULL, 18,  'Unusual Pattern',    '2026-03-01', 'Low',      'Open',          'MD account linked to two brokers with concurrent activity',  NULL);
GO

-- --------------------------------------------
-- APPROVALWORKFLOW
-- (reviewers are compliance/risk employees: 2,6,9,10,14,16,19,20)
-- --------------------------------------------
INSERT INTO dbo.ApprovalWorkflow VALUES
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
GO


-- =============================================
-- QUICK VERIFICATION QUERIES
-- =============================================

-- Row counts
SELECT 'Employee'          AS TableName, COUNT(*) AS Rows FROM dbo.Employee          UNION ALL
SELECT 'BrokerDealer',                   COUNT(*)        FROM dbo.BrokerDealer       UNION ALL
SELECT 'Account',                        COUNT(*)        FROM dbo.Account            UNION ALL
SELECT 'RestrictedSecurity',             COUNT(*)        FROM dbo.RestrictedSecurity UNION ALL
SELECT 'TradeRequest',                   COUNT(*)        FROM dbo.TradeRequest       UNION ALL
SELECT 'ComplianceAlert',                COUNT(*)        FROM dbo.ComplianceAlert    UNION ALL
SELECT 'ApprovalWorkflow',               COUNT(*)        FROM dbo.ApprovalWorkflow;

-- Requests on restricted securities (key compliance scenario)
SELECT
    tr.TradeRequestID,
    e.EmployeeName,
    e.Department,
    bd.BrokerDealerName,
    tr.SecuritySymbol,
    tr.TradeType,
    tr.Quantity,
    tr.Status,
    rs.RestrictionType,
    rs.Reason
FROM dbo.TradeRequest tr
JOIN dbo.Employee          e  ON tr.EmployeeID     = e.EmployeeID
JOIN dbo.BrokerDealer      bd ON tr.BrokerDealerID = bd.BrokerDealerID
JOIN dbo.RestrictedSecurity rs ON tr.SecuritySymbol = rs.SecuritySymbol
WHERE tr.RequestDate BETWEEN rs.StartDate AND ISNULL(rs.EndDate, '9999-12-31')
ORDER BY tr.RequestDate;

-- Alert summary by severity and broker dealer
SELECT
    bd.BrokerDealerName,
    ca.Severity,
    COUNT(*) AS AlertCount
FROM dbo.ComplianceAlert ca
JOIN dbo.TradeRequest    tr ON ca.TradeRequestID = tr.TradeRequestID
JOIN dbo.BrokerDealer    bd ON tr.BrokerDealerID = bd.BrokerDealerID
GROUP BY bd.BrokerDealerName, ca.Severity
ORDER BY bd.BrokerDealerName, ca.Severity;

-- Approval turnaround by department
SELECT
    e.Department,
    AVG(aw.TurnaroundDays) AS AvgTurnaroundDays,
    COUNT(*)               AS TotalReviewed,
    SUM(CASE WHEN aw.Decision = 'Rejected'  THEN 1 ELSE 0 END) AS Rejections,
    SUM(CASE WHEN aw.Decision = 'Escalated' THEN 1 ELSE 0 END) AS Escalations
FROM dbo.ApprovalWorkflow aw
JOIN dbo.TradeRequest     tr ON aw.TradeRequestID = tr.TradeRequestID
JOIN dbo.Employee          e ON tr.EmployeeID      = e.EmployeeID
WHERE aw.TurnaroundDays IS NOT NULL
GROUP BY e.Department
ORDER BY AvgTurnaroundDays DESC;
