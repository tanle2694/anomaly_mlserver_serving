import httpx


class MLServerClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def repository_index(self) -> set[str]:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(f"{self.base_url}/v2/repository/index", json={})
            response.raise_for_status()
            payload = response.json()
        return {item["name"] for item in payload if "name" in item}

    async def load(self, model_name: str) -> None:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(f"{self.base_url}/v2/repository/models/{model_name}/load", json={})
            response.raise_for_status()

    async def unload(self, model_name: str) -> None:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(f"{self.base_url}/v2/repository/models/{model_name}/unload", json={})
            if response.status_code not in {200, 404}:
                response.raise_for_status()

    async def infer(self, mlserver_model_name: str, values: list[list[float]]) -> dict:
        n_rows = len(values)
        n_cols = len(values[0]) if values else 0
        payload = {
            "inputs": [
                {
                    "name": "input-0",
                    "shape": [n_rows, n_cols],
                    "datatype": "FP64",
                    "data": values,
                }
            ]
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.base_url}/v2/models/{mlserver_model_name}/infer",
                json=payload,
            )
            response.raise_for_status()
            return response.json()
