#!/usr/bin/env bash
# Run the 199-task T+C suite after validating APIs, fixtures, Docker, and ports.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ $# -ne 1 ]; then
  echo "用法: ./run_batch.sh <config名>"
  echo "例如: ./run_batch.sh pangu"
  exit 2
fi

config_name=$1
config_path="model_configs/${config_name}.yaml"
tasks_dir=${TASKS_DIR:-tasks_text_only}
trials=${TRIALS:-3}
parallel=${PARALLEL:-30}
port_base_offset=${PORT_BASE_OFFSET:-12000}

if [ ! -f "$config_path" ]; then
  echo "[ERROR] 找不到 $config_path"
  exit 2
fi
if [ ! -f .env ]; then
  echo "[ERROR] 找不到普通文件 .env"
  exit 2
fi
if [ -L .env ]; then
  echo "[ERROR] .env 不允许是软链接"
  exit 2
fi
if [ ! -x .venv/bin/python ]; then
  echo "[ERROR] 找不到本仓库 .venv；请先安装依赖"
  exit 2
fi

set -a
# shellcheck disable=SC1091
source .env
set +a
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

probe_chat() {
  local label=$1 model_id=$2 base_url=$3 api_key=$4 extra_headers_json=${5:-\{\}}
  local output_file payload status response_model encoded_header decoded_header header_name
  local has_authorization=0 has_content_type=0
  local -a header_args
  output_file=$(mktemp)
  payload=$(jq -nc --arg model "$model_id" '{model:$model,messages:[{role:"user",content:"Reply OK"}],max_tokens:8}')
  header_args=()
  while IFS= read -r encoded_header; do
    [ -n "$encoded_header" ] || continue
    decoded_header=$(printf '%s' "$encoded_header" | base64 --decode)
    header_name=${decoded_header%%:*}
    case ${header_name,,} in
      authorization) has_authorization=1 ;;
      content-type) has_content_type=1 ;;
    esac
    header_args+=(-H "$decoded_header")
  done < <(
    jq -r 'to_entries[] | "\(.key): \(.value)" | @base64' \
      <<<"$extra_headers_json"
  )
  if [ "$has_authorization" -eq 0 ]; then
    header_args+=(-H "Authorization: Bearer $api_key")
  fi
  if [ "$has_content_type" -eq 0 ]; then
    header_args+=(-H 'Content-Type: application/json')
  fi
  status=$(curl -sS -m 45 -o "$output_file" -w '%{http_code}' \
    "${base_url%/}/chat/completions" \
    "${header_args[@]}" \
    -d "$payload" || true)
  if [ "$status" = 200 ]; then
    if response_model=$(.venv/bin/python - "$output_file" <<'PY'
import json, pathlib, sys
try:
    data = json.loads(pathlib.Path(sys.argv[1]).read_text(errors="replace"))
    choices = data.get("choices")
    assert isinstance(choices, list) and choices
    assert isinstance(choices[0].get("message"), dict)
    print(data.get("model") or "<未返回 model 字段>")
except Exception as exc:
    print(f"响应不是有效的 chat completion：{exc}", file=sys.stderr)
    raise SystemExit(1)
PY
    ); then
      echo "  [OK] $label：请求=$model_id，响应=$response_model"
      rm -f "$output_file"
      return 0
    fi
    echo "  [FAIL] $label：HTTP 200，但响应格式无效"
    rm -f "$output_file"
    return 1
  fi
  echo "  [FAIL] $label：$model_id，HTTP ${status:-请求失败}"
  .venv/bin/python - "$output_file" <<'PY'
import json, pathlib, sys
text = pathlib.Path(sys.argv[1]).read_text(errors="replace")[:1000]
try:
    data = json.loads(text)
    print("         " + json.dumps(data, ensure_ascii=False)[:800])
except Exception:
    print("         " + text.replace("\n", " ")[:800])
PY
  rm -f "$output_file"
  return 1
}

preflight() {
  local failed=0 task_count
  echo "=========== 起跑前自检 ==========="
  echo "  config=$config_name tasks=$tasks_dir trials=$trials parallel=$parallel port_base_offset=$port_base_offset"

  if .venv/bin/python -c 'import fastapi, uvicorn, pypdf, docker' >/dev/null 2>&1; then
    echo "  [OK] Python/mock/sandbox 依赖完整"
  else
    echo "  [FAIL] 依赖不完整：uv pip install --python .venv/bin/python -r requirements.txt"
    failed=1
  fi

  task_count=$(find -L "$tasks_dir" -mindepth 2 -maxdepth 2 -name task.yaml 2>/dev/null | wc -l)
  if [ "$task_count" -eq 199 ]; then
    echo "  [OK] 任务集完整：199（T=161，C=38）"
  else
    echo "  [FAIL] $tasks_dir 应有 199 道，当前为 $task_count"
    failed=1
  fi

  local fixture missing=0
  local fixtures=(
    "tasks/T091_pinbench_humanize_blog/fixtures/docs/ai_blog.txt"
    "tasks/T096_pinbench_business_metrics_summary/fixtures/docs/quarterly_sales.csv"
    "tasks/T096_pinbench_business_metrics_summary/fixtures/docs/company_expenses.xlsx"
    "tasks/T096_pinbench_business_metrics_summary/fixtures/docs/company_expenses_extracted.txt"
    "tasks/T097_pinbench_eli5_model_summary/fixtures/docs/GPT4.pdf"
    "tasks/T098_pinbench_openclaw_facts/fixtures/docs/OpenClaw Agent Use Cases and Gap Analysis for PinchBench.pdf"
  )
  for fixture in "${fixtures[@]}"; do
    if [ ! -s "$fixture" ]; then
      echo "  [FAIL] 缺少 fixture：$fixture"
      missing=1
    fi
  done
  if [ "$missing" -eq 0 ]; then
    echo "  [OK] 额外 fixtures 完整"
  else
    failed=1
  fi

  local config_json model_id model_url model_key model_headers
  local judge_id judge_url judge_key ua_id ua_url ua_key sandbox_on sandbox_image
  config_json=$(.venv/bin/python - "$config_path" <<'PY'
import json, sys
from claw_eval.config import load_config
c = load_config(sys.argv[1])
print(json.dumps({
    "model": {
        "id": c.model.model_id,
        "url": c.model.base_url or "",
        "key": c.model.api_key or "",
        "headers": c.model.extra_headers or {},
    },
    "judge": {
        "id": c.judge.model_id,
        "url": c.judge.base_url or "",
        "key": c.judge.api_key or "",
    },
    "user_agent": {
        "id": c.user_agent_model.model_id,
        "url": c.user_agent_model.base_url or "",
        "key": c.user_agent_model.api_key or "",
    },
    "sandbox": {
        "enabled": c.sandbox.enabled,
        "image": c.sandbox.image,
    },
}))
PY
  )
  model_id=$(jq -r '.model.id' <<<"$config_json")
  model_url=$(jq -r '.model.url' <<<"$config_json")
  model_key=$(jq -r '.model.key' <<<"$config_json")
  model_headers=$(jq -c '.model.headers' <<<"$config_json")
  judge_id=$(jq -r '.judge.id' <<<"$config_json")
  judge_url=$(jq -r '.judge.url' <<<"$config_json")
  judge_key=$(jq -r '.judge.key' <<<"$config_json")
  ua_id=$(jq -r '.user_agent.id' <<<"$config_json")
  ua_url=$(jq -r '.user_agent.url' <<<"$config_json")
  ua_key=$(jq -r '.user_agent.key' <<<"$config_json")
  sandbox_on=$(jq -r 'if .sandbox.enabled then "1" else "0" end' <<<"$config_json")
  sandbox_image=$(jq -r '.sandbox.image' <<<"$config_json")

  probe_chat "被测模型" "$model_id" "$model_url" "$model_key" "$model_headers" || failed=1
  if [ "$model_url|$model_key" = "$judge_url|$judge_key" ]; then
    echo "  [WARN] 被测模型与 judge 共用 endpoint/key；会共享限流与额度"
  fi
  if probe_chat "judge" "$judge_id" "$judge_url" "$judge_key"; then
    if [ "$ua_id|$ua_url|$ua_key" = "$judge_id|$judge_url|$judge_key" ]; then
      echo "  [OK] user-agent 共用已验证链路：$ua_id"
    else
      probe_chat "user-agent" "$ua_id" "$ua_url" "$ua_key" || failed=1
    fi
  else
    failed=1
  fi

  if grep -rlq 'web_real' "$tasks_dir"/*/task.yaml 2>/dev/null; then
    local serp_url=${SERP_API_URL:-https://yibuapi.com/serper/search}
    local serp_key=${SERP_API_KEY:-${SERP_DEV_KEY:-${YIBUAPI_KEY:-}}}
    local serp_body
    if [ -z "$serp_key" ]; then
      echo "  [FAIL] 缺少 SERP_API_KEY / SERP_DEV_KEY"
      failed=1
    elif serp_body=$(curl -fsS -m 30 "$serp_url" \
      -H "Authorization: Bearer $serp_key" \
      -H 'Content-Type: application/json' \
      -d '{"q":"OpenAI official documentation","num":1,"gl":"us","hl":"en","page":1}'); then
      if printf '%s' "$serp_body" | .venv/bin/python -c 'import json,sys; d=json.load(sys.stdin); assert isinstance(d.get("organic"), list) and d["organic"]'; then
        echo "  [OK] SERP：$serp_url"
      else
        echo "  [FAIL] SERP 返回格式异常或没有搜索结果：$serp_url"
        failed=1
      fi
    else
      echo "  [FAIL] SERP 调用失败：$serp_url"
      failed=1
    fi
  fi

  if [ "$sandbox_on" != 1 ]; then
    echo "  [FAIL] sandbox.enabled=false；/workspace 文件题会失败"
    failed=1
  elif ! docker ps >/dev/null 2>&1; then
    echo "  [FAIL] Docker 不可用"
    failed=1
  elif ! docker image inspect "$sandbox_image" >/dev/null 2>&1; then
    echo "  [FAIL] 缺少镜像 $sandbox_image；先运行 claw-eval build-image"
    failed=1
  else
    local memory_slots cpu_slots
    memory_slots=$(free -g | awk 'NR==2 {print int($7/4)}')
    cpu_slots=$(( $(nproc) / 2 ))
    if [ "$parallel" -gt "$memory_slots" ] || [ "$parallel" -gt "$cpu_slots" ]; then
      echo "  [FAIL] parallel=$parallel 超出资源：内存槽=$memory_slots CPU槽=$cpu_slots"
      failed=1
    else
      echo "  [OK] sandbox：$sandbox_image，资源支持 $parallel 并发"
    fi
  fi

  if ! .venv/bin/python - "$port_base_offset" "$parallel" "$tasks_dir" <<'PY'
import glob, re, subprocess, sys, yaml
offset, parallel, task_dir = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
busy = set()
for line in subprocess.run(["ss", "-ltnH"], capture_output=True, text=True, check=True).stdout.splitlines():
    fields = line.split()
    if len(fields) >= 4 and (match := re.search(r":(\d+)$", fields[3])):
        busy.add(int(match.group(1)))
bases = {
    int(service["port"])
    for path in glob.glob(f"{task_dir}/*/task.yaml")
    for service in ((yaml.safe_load(open(path, encoding="utf-8")) or {}).get("services") or [])
    if service.get("port")
}
required = {base + offset + slot * 50 for base in bases for slot in range(parallel)}
hits = sorted(required & busy)
if hits:
    print(f"  [FAIL] 端口冲突：{hits}")
    raise SystemExit(1)
print(f"  [OK] 端口段 {min(required)}-{max(required)} 空闲")
PY
  then
    failed=1
  fi

  local orphan_count
  orphan_count=$(pgrep -f "$PWD/.venv/bin/python.*mock_services/" 2>/dev/null | wc -l || true)
  if [ "$orphan_count" -gt 0 ]; then
    # Mock services are expected while another batch is running. Its occupied
    # ports were already checked above, so a disjoint offset is safe.
    if pgrep -f 'claw_eval[.]cli batch' >/dev/null 2>&1; then
      echo "  [OK] 检测到其他运行中 batch（$orphan_count 个 mock service）；端口段不冲突"
    else
      echo "  [FAIL] 有 $orphan_count 个残留 mock service，且没有运行中的 batch；先运行："
      echo "         pgrep -f '$PWD/.venv/bin/python.*mock_services/' | xargs -r kill"
      failed=1
    fi
  else
    echo "  [OK] 无残留 mock service"
  fi

  return "$failed"
}

if ! preflight; then
  echo "自检未通过，已中止，不会启动评测。"
  exit 1
fi
if [ -n "${PREFLIGHT_ONLY:-}" ]; then
  echo "自检通过；PREFLIGHT_ONLY 已设置，不启动评测。"
  exit 0
fi

mkdir -p logs
stamp=$(date +%m%d_%H%M)
echo "=========== $config_name 开始 $(date '+%F %T') ==========="
.venv/bin/python -m claw_eval.cli batch \
  --tasks-dir "$tasks_dir" \
  --config "$config_path" \
  --parallel "$parallel" \
  --trials "$trials" \
  --port-base-offset "$port_base_offset" \
  2>&1 | tee "logs/batch_${config_name}_${stamp}.log"
echo "=========== $config_name 完成 $(date '+%F %T') ==========="
