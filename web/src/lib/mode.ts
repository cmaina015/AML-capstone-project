/**
 * Where predictions run, fixed at build time:
 *  - "browser" (default): ONNX Runtime Web in the visitor's browser; the site is fully static.
 *  - "server": POST /api/predict to the FastAPI + PyTorch backend (Docker build, NEXT_PUBLIC_INFERENCE=server).
 */
export const SERVER_MODE = process.env.NEXT_PUBLIC_INFERENCE === "server";
