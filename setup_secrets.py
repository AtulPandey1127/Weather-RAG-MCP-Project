"""
One-time Databricks secret setup.

Only the Lakebase connection URL is stored here.
Weather APIs used by this project do not require a paid API key.

Usage:
    python setup_secrets.py

Never commit secret values to source control.
"""

from getpass import getpass

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace


SCOPE = "database"
KEY = "lakebase-url"


def main() -> None:
    workspace_client = WorkspaceClient()

    print("Creating/updating Lakebase secret...")
    print("The value will not be displayed.")

    lakebase_url = getpass(
        "Paste your Lakebase PostgreSQL connection URL: "
    )

    if not lakebase_url:
        raise ValueError("Lakebase URL cannot be empty.")

    try:
        workspace_client.secrets.create_scope(scope=SCOPE)
        print(f"Created secret scope: {SCOPE}")
    except Exception:
        print(f"Using existing secret scope: {SCOPE}")

    workspace_client.secrets.put_secret(
        scope=SCOPE,
        key=KEY,
        string_value=lakebase_url,
    )

    workspace_client.secrets.put_acl(
        scope=SCOPE,
        principal="users",
        permission=workspace.AclPermission.READ,
    )

    print("Lakebase secret configured successfully.")


if __name__ == "__main__":
    main()
