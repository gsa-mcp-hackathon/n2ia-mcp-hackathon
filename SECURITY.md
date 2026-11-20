# Security Configuration

This document summarizes the security-related configuration used for **N2IA Federal Spending Intelligence**, a hackathon solution deployed on Databricks. It describes the controls reflected in the project implementation and configuration, rather than a general security policy or a claim of production authorization.

## Deployed components and identities

| Component | Configured security approach |
| --- | --- |
| Streamlit client | Deployed as a Databricks App with an application service principal for service-to-service calls. |
| Databricks Supervisor Agent | Exposed through a Databricks serving endpoint; the client app's service principal was granted **CAN QUERY** on that endpoint. |
| MCP server | Deployed as a separate Databricks App and registered with the Supervisor Agent as an MCP tool source. |
| Databricks resources | Application resource configuration and associated permissions were used to enable the client to reach the Supervisor endpoint and the server to use its required Databricks resources. |

## Agent and MCP connectivity

The application follows this invocation path:

```text
User -> Streamlit Databricks App -> Supervisor Agent endpoint
                                      |
                                      v
                                MCP Server App
                                      |
                                      v
                        USAspending.gov / Databricks resources
```

- The Streamlit App was configured with the Supervisor Agent endpoint as an application resource.
- The client application's service principal was granted **CAN QUERY** permission on the Supervisor endpoint. Both the endpoint permission and application resource registration were needed for successful invocation.
- The Supervisor Agent was configured to invoke tools exposed by the separate MCP Server App; analytical calculations execute in the MCP server rather than in the client or the LLM.
- This configuration describes the verified client-to-Supervisor access path. It does **not** independently establish that the MCP App's HTTP endpoint has its own authentication enforcement; that must be checked in the deployed App settings before making a broader claim.

## MCP Server App service principal: Unity Catalog and SQL access

The **MCP Server Databricks App runs under its own application service principal**. We configured access for that identity to the resources required by the analytical tools; this is separate from the Streamlit client's permission to query the Supervisor Agent.

| Resource | Why the MCP server needs it | Relevant permissions to verify in Databricks |
| --- | --- | --- |
| Unity Catalog Volume containing `all-MiniLM-L6-v2` | Load the locally stored Hugging Face model for semantic similarity analysis | `USE CATALOG`, `USE SCHEMA`, `READ VOLUME` |
| Unity Catalog Volume containing agency/category JSON | Read configuration-driven agency and service-category choices | `USE CATALOG`, `USE SCHEMA`, `READ VOLUME` |
| Delta cache tables | Read reusable award snapshots and persist new snapshots | `USE CATALOG`, `USE SCHEMA`, `SELECT`, `MODIFY` on the relevant tables; `CREATE TABLE` only if the application creates tables |
| Databricks SQL warehouse | Execute SQL operations used by the Delta-backed cache | `CAN USE` on the configured SQL warehouse |

**Configuration record:** Access to the model/configuration assets and Delta cache was part of the MCP server setup. The table above lists the *privileges normally needed for those operations*, not an independently verified export of the exact grants applied. Confirm the catalog, schema, Volume, table names, service-principal identity, and actual privileges in the Databricks workspace before treating this as an access audit.

The MCP server uses the configured `N2IA_SQL_WAREHOUSE_ID` for SQL warehouse connectivity. The application identity needs both permission to use that warehouse and the applicable Unity Catalog object permissions; warehouse access alone does not grant access to tables.

### How to verify the actual grants

In Databricks, open the MCP Server App and identify its **service principal**. Then inspect **Permissions** for each relevant Volume, schema/table, and SQL warehouse. For Unity Catalog, the following queries can help confirm grants (replace placeholders with actual three-part names):

```sql
SHOW GRANTS ON VOLUME <catalog>.<schema>.<volume>;
SHOW GRANTS ON TABLE <catalog>.<schema>.<cache_table>;
SHOW GRANTS ON SCHEMA <catalog>.<schema>;
SHOW GRANTS ON CATALOG <catalog>;
```

Check that the principal shown in the results is the **MCP Server App service principal**, not the developer's user account or the Streamlit App service principal. Review the SQL warehouse's **Permissions** separately.

## Client App and Supervisor endpoint permissions

- The **Streamlit Client App service principal** was granted **`CAN QUERY`** on the Supervisor Agent serving endpoint.
- The Supervisor endpoint was also added as a **resource of the Streamlit App**; endpoint permission and resource registration were both needed.
- The **Supervisor Agent** was configured with the separately deployed **MCP Server Databricks App** as its tool source.
- These are distinct authorization relationships: granting the client access to the Supervisor does **not** grant the MCP Server access to Unity Catalog objects.

## Source data and credentials

- Contract award information is retrieved from the **public USAspending.gov API**; that source does not require an API key for the demonstrated workflow.
- The application uses Databricks-managed service identities and resource permissions for its internal interactions, rather than embedding Supervisor endpoint credentials in the UI.
- The repository should not contain secrets, tokens, or local credential files. The specific secret-store settings and repository scanning controls have not been independently verified here.

## Analytical controls

- The MCP server uses bounded queries and reports retrieval coverage and limitations.
- Contract matching is calculated by the MCP server using lexical and semantic methods plus service-function and performance-period evidence; the Supervisor Agent interprets the structured output.
- Similar contract relationships are presented as **candidates for human acquisition review**, not confirmed duplication, waste, or savings.

## Scope of this document

This is a record of the hackathon's configured security architecture, **not** a vulnerability disclosure policy, a security audit, an ATO, or a claim of production readiness. Before broader deployment, verify MCP endpoint authentication, the exact service-principal and Unity Catalog grants, logging access, and any agency-specific requirements.
