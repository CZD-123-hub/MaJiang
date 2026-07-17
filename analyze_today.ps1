# 九江麻将对局分析脚本 - 2026-07-16

$lines = Get-Content "d:\MaJiang\temp_today_games.jsonl"
$allGames = $lines | ForEach-Object { $_ | ConvertFrom-Json }

Write-Output "=========================================="
Write-Output "     九江麻将对局分析报告"
Write-Output "     日期: 2026-07-16"
Write-Output "=========================================="
Write-Output ""

# 基本统计
Write-Output "【基本信息】"
Write-Output "  总对局数: $($allGames.Count) 局"
Write-Output "  不同房间数: $($allGames | Select-Object -ExpandProperty data | Select-Object -ExpandProperty room_id -Unique | Measure-Object).Count"
$startTime = ([datetime]$allGames[0].timestamp).ToString("HH:mm:ss")
$endTime = ([datetime]$allGames[-1].timestamp).ToString("HH:mm:ss")
Write-Output "  时间跨度: $startTime - $endTime"
Write-Output ""

# 胜利方式统计
$winTypes = @{}
$winnerCount = @{}
$totalScoreChange = @{0=0.0; 1=0.0; 2=0.0; 3=0.0}
$bankerCount = @{}

foreach ($game in $allGames) {
    # 胜利方式
    $wt = $game.data.win_type
    if ($wt) {
        if (-not $winTypes.ContainsKey($wt)) { $winTypes[$wt] = 0 }
        $winTypes[$wt]++
    }

    # 赢家统计
    $winner = $game.data.winner
    if ($null -ne $winner) {
        if (-not $winnerCount.ContainsKey($winner)) { $winnerCount[$winner] = 0 }
        $winnerCount[$winner]++
    }

    # 得分累计
    for ($i=0; $i -lt 4; $i++) {
        if ($game.data.scores -and $game.data.scores.Count -gt $i) {
            $totalScoreChange[$i] += $game.data.scores[$i]
        }
    }

    # 庄家统计
    $banker = $game.data.banker_position
    if ($null -ne $banker) {
        if (-not $bankerCount.ContainsKey($banker)) { $bankerCount[$banker] = 0 }
        $bankerCount[$banker]++
    }
}

Write-Output "【胜利方式】"
$winTypes.GetEnumerator() | Sort-Object Name | ForEach-Object {
    $pct = [math]::Round($_.Value/$allGames.Count*100, 1)
    Write-Output "  $($_.Key): $($_.Value) 局 ($pct%)"
}
Write-Output ""

Write-Output "【各位置表现】"
Write-Output "位置 | 胜场 | 胜率   | 总得分 | 平均得分"
Write-Output "---- | ---- | ------ | ------ | --------"
0..3 | ForEach-Object {
    $pos = $_
    $wins = if ($winnerCount.ContainsKey($pos)) { $winnerCount[$pos] } else { 0 }
    $winRate = [math]::Round($wins / $allGames.Count * 100, 1)
    $totalScore = $totalScoreChange[$pos]
    $avgScore = [math]::Round($totalScore / $allGames.Count, 2)
    Write-Output ("{0,4} | {1,4} | {2,5}% | {3,6} | {4,8}" -f $pos, $wins, $winRate, $totalScore, $avgScore)
}
Write-Output ""

Write-Output "【庄家位置分布】"
$bankerCount.GetEnumerator() | Sort-Object {[int]$_.Name} | ForEach-Object {
    $pct = [math]::Round($_.Value/$allGames.Count*100, 1)
    Write-Output "  位置 $($_.Name): $($_.Value) 次 ($pct%)"
}
Write-Output ""

Write-Output "【得分排名】"
$rankings = 0..3 | Sort-Object {$totalScoreChange[$_]} -Descending
for ($i=0; $i -lt 4; $i++) {
    $pos = $rankings[$i]
    $score = $totalScoreChange[$pos]
    Write-Output "  第$($i+1)名: 位置$pos ($score 分)"
}
Write-Output ""

Write-Output "【关键发现】"
# 找出表现最好和最差的位置
$bestPos = $rankings[0]
$worstPos = $rankings[3]
Write-Output "  • 表现最佳: 位置$bestPos (+$($totalScoreChange[$bestPos])分, $($winnerCount[$bestPos])胜)"
Write-Output "  • 表现最差: 位置$worstPos ($($totalScoreChange[$worstPos])分, $(if($winnerCount.ContainsKey($worstPos)){$winnerCount[$worstPos]}else{0})胜)"
Write-Output "  • 全部自摸，无点炮记录"

# 庄家优势分析
$bankerWins = 0
foreach ($game in $allGames) {
    if ($game.data.banker_position -eq $game.data.winner) {
        $bankerWins++
    }
}
$bankerWinRate = [math]::Round($bankerWins / $allGames.Count * 100, 1)
Write-Output "  • 庄家胜率: $bankerWinRate% ($bankerWins/$($allGames.Count))"
Write-Output ""
