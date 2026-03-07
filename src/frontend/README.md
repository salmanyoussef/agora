# Agora frontend

Sample UI for the Agora Q&A pipeline over French open data.

## Run

1. Start the backend from `src/backend`:
   ```bash
   uvicorn app.main:app --reload
   ```
2. Open **http://localhost:8000/** in a browser. The backend serves this frontend at `/`.

## API base

The page uses the current origin as the API base. To point to another host/port, add in `<head>`:

```html
<meta name="api-base" content="http://localhost:8000" />
```
