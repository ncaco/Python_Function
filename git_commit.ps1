# UTF-8 인코딩 설정
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$PSDefaultParameterValues['*:Encoding'] = 'utf8'

$message = $args[0]

if ([string]::IsNullOrEmpty($message)) {
    Write-Output "커밋 메시지를 입력해주세요."
    Write-Output "사용법: .\git_commit.ps1 '커밋 메시지'"
    exit 1
}

git add .
git status
git commit -m "$message"
git push origin main -u

Write-Host "Git Commit and Push Completed!" -ForegroundColor Green