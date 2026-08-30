[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^[A-Za-z0-9_-]{4,100}$')]
    [string]$EndpointId,
    [string]$RequestFile = (Join-Path $PSScriptRoot 'zakul_runpod\examples\generate-20s.json'),
    [ValidatePattern('^[A-Za-z0-9_-]{1,150}$')]
    [string]$JobId,
    [ValidateRange(60,14400)]
    [int]$TimeoutSeconds = 2100,
    [string]$OutputDirectory = (Join-Path $PSScriptRoot 'runpod-results')
)

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$token = $env:RUNPOD_API_KEY
if ([string]::IsNullOrWhiteSpace($token)) {
    $secureToken = Read-Host 'RunPod API key (hidden; not saved)' -AsSecureString
    $tokenPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
    try { $token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($tokenPointer) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($tokenPointer) }
}
if ([string]::IsNullOrWhiteSpace($token)) { throw 'A RunPod API key is required.' }
$headers = @{ Authorization = "Bearer $($token.Trim())" }
$endpoint = "https://api.runpod.ai/v2/$EndpointId"

try {
    if (-not $JobId) {
        $body = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $RequestFile).Path)
        $null = $body | ConvertFrom-Json
        Write-Host 'Submitting ONE job. This starts a paid RunPod test.'
        try {
            $submitted = Invoke-RestMethod -Method Post -Uri "$endpoint/run" -Headers $headers -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 60
        } catch {
            throw 'Submission was not confirmed. Check RunPod Requests before retrying: the job may already be running. Do not submit again blindly.'
        }
        $JobId = [string]$submitted.id
        if ($JobId -notmatch '^[A-Za-z0-9_-]{1,150}$') { throw 'RunPod returned no valid job ID.' }
    }
    Write-Host "Job ID: $JobId"
    Write-Host 'Ctrl+C stops this client, NOT the cloud job. Cancel the job in RunPod if needed.'
    Write-Host "Resume polling this job without submitting a new one: .\test-runpod.ps1 -EndpointId $EndpointId -JobId $JobId"
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    $lastStatus = ''
    $transientErrors = 0
    while ([DateTime]::UtcNow -lt $deadline) {
        try {
            $state = Invoke-RestMethod -Method Get -Uri "$endpoint/status/$JobId" -Headers $headers -TimeoutSec 60
            $transientErrors = 0
        } catch {
            $httpStatus = 0
            if ($_.Exception.Response) { $httpStatus = [int]$_.Exception.Response.StatusCode }
            if ($httpStatus -ne 0 -and $httpStatus -notin @(408,429,500,502,503,504)) {
                throw "Status check failed with HTTP $httpStatus. Check the key, endpoint and job ID."
            }
            $transientErrors++
            if ($transientErrors -ge 12) { throw 'Repeated status errors. Resume later with -JobId; do not resubmit.' }
            Start-Sleep -Seconds ([Math]::Min(30, $transientErrors * 3))
            continue
        }
        $status = [string]$state.status
        if ($status -ne $lastStatus) { Write-Host "Status: $status"; $lastStatus = $status }
        if ($status -eq 'COMPLETED') {
            if ($state.output.error) { throw [string]$state.output.error }
            $suffix = [Guid]::NewGuid().ToString('N').Substring(0,8)
            $destination = Join-Path $OutputDirectory ((Get-Date -Format 'yyyyMMdd-HHmmss') + "-$suffix")
            $null = [IO.Directory]::CreateDirectory($destination)
            $trackNumber = 0
            foreach ($track in $state.output.tracks) {
                $trackNumber++
                if ($track.mp3.base64) {
                    $bytes = [Convert]::FromBase64String([string]$track.mp3.base64)
                    if ($bytes.Length -gt 5000000) { throw 'Unexpectedly large inline MP3.' }
                    $path = Join-Path $destination "take-$trackNumber.mp3"
                    [IO.File]::WriteAllBytes($path, $bytes)
                    $track.mp3.PSObject.Properties.Remove('base64')
                    Write-Host "Saved MP3: $path"
                }
            }
            [IO.File]::WriteAllText((Join-Path $destination 'result.json'), ($state | ConvertTo-Json -Depth 30), [Text.UTF8Encoding]::new($false))
            Write-Host "Result saved: $destination"
            if ($state.output.operation -ne 'generate') { $state.output | Format-List }
            return
        }
        if ($status -in @('FAILED','CANCELLED','TIMED_OUT')) {
            throw "RunPod job $status. $($state.error)"
        }
        if ($status -notin @('IN_QUEUE','IN_PROGRESS')) { throw "Unknown RunPod status: $status" }
        Start-Sleep -Seconds 5
    }
    throw 'Client timeout. The cloud job may still be running. Check RunPod or resume with -JobId.'
} finally {
    $headers.Clear()
    $token = $null
}
