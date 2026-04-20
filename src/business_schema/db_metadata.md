# Table Metadata

Describes every table, column, allowed values, and foreign key relationships in the database.

---

## Entity Relationship Overview

```
Employee ──────────────── Account ──────── BrokerDealer
    │                        │
    │                        │
    ├──── TradeRequest ───────┘
    │         │
    │         ├──── ComplianceAlert
    │         │
    │         └──── ApprovalWorkflow ──── Employee (as Reviewer)
    │
    └──── ComplianceAlert (direct, no trade)
```

---

## Tables

### Employee
Represents individuals in the organisation. Used as the subject of trade requests, alerts, and workflow reviews.

| Column | Type | Description |
|---|---|---|
| `EmployeeID` | INTEGER PK | Unique employee identifier |
| `EmployeeName` | TEXT | Full name |
| `Department` | TEXT | Department (see values below) |
| `JobTitle` | TEXT | Job title (Analyst, Trader, VP, MD, Director, etc.) |
| `HireDate` | TEXT | ISO date of hire (`YYYY-MM-DD`) |
| `Status` | TEXT | `Active` or `Terminated` |
| `UpdatedAt` | TEXT | Last modified timestamp — auto-set on insert |

**Department values:** `Investment Banking`, `Compliance`, `Technology`, `Trading`, `Risk Management`, `Legal`, `Operations`

---

### BrokerDealer
External brokers through which employees route trade requests.

| Column | Type | Description |
|---|---|---|
| `BrokerDealerID` | INTEGER PK | Unique broker identifier |
| `BrokerDealerName` | TEXT | Broker name (e.g. Fidelity, Morgan Stanley) |
| `Country` | TEXT | Country of registration (`USA`, `UK`, `DE`) |
| `RegistrationNumber` | TEXT | Regulatory registration code |
| `UpdatedAt` | TEXT | Last modified timestamp |

---

### Account
Links an employee to a broker dealer. An employee can have multiple accounts at different brokers.

| Column | Type | Description |
|---|---|---|
| `AccountID` | INTEGER PK | Unique account identifier |
| `EmployeeID` | INTEGER FK → Employee | Account owner |
| `BrokerDealerID` | INTEGER FK → BrokerDealer | Broker holding the account |
| `AccountNumber` | TEXT | Account reference number |
| `AccountType` | TEXT | `Individual`, `Retirement`, `Joint` |
| `OpenDate` | TEXT | ISO date account was opened |
| `Status` | TEXT | `Active` or `Closed` |
| `UpdatedAt` | TEXT | Last modified timestamp |

---

### RestrictedSecurity
Securities placed under trading restrictions. A trade request on a restricted symbol triggers a violation.

| Column | Type | Description |
|---|---|---|
| `RestrictionID` | INTEGER PK | Unique restriction identifier |
| `SecuritySymbol` | TEXT | Stock ticker (e.g. `AAPL`, `MSFT`, `TSLA`) |
| `RestrictionType` | TEXT | Type of restriction (see values below) |
| `StartDate` | TEXT | ISO date restriction begins |
| `EndDate` | TEXT | ISO date restriction ends — `NULL` means still active |
| `Reason` | TEXT | Plain-English reason for the restriction |
| `AddedBy` | TEXT | Name of person who added the restriction |
| `UpdatedAt` | TEXT | Last modified timestamp |

**RestrictionType values:**
- `Insider List` — security is on the insider trading watch list
- `Blackout` — time-bound trading blackout period (EndDate is set)
- `Watch List` — under regulatory or compliance monitoring

**Active restriction check:** `StartDate <= date('now') AND (EndDate IS NULL OR EndDate >= date('now'))`

---

### TradeRequest
A request by an employee to buy or sell a security through a broker dealer.

| Column | Type | Description |
|---|---|---|
| `TradeRequestID` | INTEGER PK | Unique trade request identifier |
| `EmployeeID` | INTEGER FK → Employee | Employee making the request |
| `BrokerDealerID` | INTEGER FK → BrokerDealer | Broker to execute through |
| `RequestDate` | TEXT | ISO date of request |
| `SecuritySymbol` | TEXT | Stock ticker |
| `TradeType` | TEXT | `BUY` or `SELL` |
| `Quantity` | INTEGER | Number of units requested |
| `Status` | TEXT | Current status (see values below) |
| `UpdatedAt` | TEXT | Last modified timestamp |

**Status values:**
- `Pending` — awaiting review
- `Approved` — cleared by compliance
- `Rejected` — denied (usually due to restriction or policy breach)
- `Escalated` — escalated to senior compliance or legal

---

### ComplianceAlert
An alert raised against an employee, optionally linked to a specific trade request. Alerts can be raised directly (no trade) for behavioural patterns.

| Column | Type | Description |
|---|---|---|
| `AlertID` | INTEGER PK | Unique alert identifier |
| `TradeRequestID` | INTEGER FK → TradeRequest | Related trade — `NULL` for behaviour-only alerts |
| `EmployeeID` | INTEGER FK → Employee | Employee the alert is raised against |
| `AlertType` | TEXT | Type of alert (see values below) |
| `AlertDate` | TEXT | ISO date the alert was raised |
| `Severity` | TEXT | `Low`, `Medium`, `High`, `Critical` |
| `Status` | TEXT | Current status (see values below) |
| `Description` | TEXT | Plain-English description of the alert |
| `ResolvedDate` | TEXT | ISO date resolved — `NULL` if still open |
| `UpdatedAt` | TEXT | Last modified timestamp |

**AlertType values:**
- `Restricted Security` — trade attempted on a restricted symbol
- `Insider Trading` — suspected insider trading activity
- `Excessive Volume` — trade quantity exceeds policy threshold
- `Unusual Pattern` — behavioural anomaly (off-hours, multi-broker, repeat pattern)

**Status values:**
- `Open` — newly raised, not yet actioned
- `Investigating` — under active compliance review
- `Escalated` — escalated to legal or senior management
- `Closed` — resolved and closed

---

### ApprovalWorkflow
Records the review decision for a trade request. Each trade request has one workflow record.

| Column | Type | Description |
|---|---|---|
| `WorkflowID` | INTEGER PK | Unique workflow record identifier |
| `TradeRequestID` | INTEGER FK → TradeRequest | The trade being reviewed |
| `ReviewerID` | INTEGER FK → Employee | Employee who performed the review |
| `ReviewDate` | TEXT | ISO date the review was completed — `NULL` if pending |
| `Decision` | TEXT | Review outcome (see values below) |
| `Comments` | TEXT | Reviewer notes |
| `TurnaroundDays` | INTEGER | Days between RequestDate and ReviewDate — `NULL` if pending |
| `UpdatedAt` | TEXT | Last modified timestamp |

**Decision values:**
- `Approved` — trade cleared
- `Rejected` — trade denied
- `Escalated` — referred to senior review
- `Pending` — review not yet completed

---

## Foreign Key Summary

| Table | Column | References |
|---|---|---|
| Account | EmployeeID | Employee.EmployeeID |
| Account | BrokerDealerID | BrokerDealer.BrokerDealerID |
| TradeRequest | EmployeeID | Employee.EmployeeID |
| TradeRequest | BrokerDealerID | BrokerDealer.BrokerDealerID |
| ComplianceAlert | TradeRequestID | TradeRequest.TradeRequestID (nullable) |
| ComplianceAlert | EmployeeID | Employee.EmployeeID |
| ApprovalWorkflow | TradeRequestID | TradeRequest.TradeRequestID |
| ApprovalWorkflow | ReviewerID | Employee.EmployeeID |

---

## Common Join Patterns

```sql
-- Employee → their trade requests
Employee e JOIN TradeRequest tr ON e.EmployeeID = tr.EmployeeID

-- Trade request → its compliance alert
TradeRequest tr JOIN ComplianceAlert ca ON tr.TradeRequestID = ca.TradeRequestID

-- Trade request → its approval decision
TradeRequest tr JOIN ApprovalWorkflow aw ON tr.TradeRequestID = aw.TradeRequestID

-- Employee → their accounts → broker
Employee e JOIN Account a ON e.EmployeeID = a.EmployeeID
           JOIN BrokerDealer bd ON a.BrokerDealerID = bd.BrokerDealerID

-- Violations: trade on an active restricted security
TradeRequest tr JOIN RestrictedSecurity rs
    ON tr.SecuritySymbol = rs.SecuritySymbol
   AND tr.RequestDate BETWEEN rs.StartDate AND COALESCE(rs.EndDate, '9999-12-31')
```

---

## Notes

- All tables include an `UpdatedAt` column set to `datetime('now')` on insert. This column is used by the data fingerprint system to detect row-level changes between refresh cycles.
- `ComplianceAlert.TradeRequestID` is nullable — alerts can be raised directly against an employee for behavioural patterns with no associated trade.
- `ApprovalWorkflow.ReviewDate` and `TurnaroundDays` are `NULL` while a review is `Pending`.
- `RestrictedSecurity.EndDate = NULL` means the restriction is still active with no defined end date.
