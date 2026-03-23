function sensitivity_analysis_from_excel(userCfg)
clc; close all;

cfg = default_config();
if nargin >= 1 && ~isempty(userCfg)
    cfg = merge_config(cfg, userCfg);
end
prepare_output_root(cfg.outputRoot, cfg.cleanOutputRoot);
addpath(genpath(cfg.matlabSourceDir));
apply_graphics_defaults(cfg);

T = readtable(cfg.excelPath, 'PreserveVariableNames', true);
requiredCols = {'times','ia','phi1','phi2','phi3','rad'};
for k = 1:numel(requiredCols)
    if ~ismember(requiredCols{k}, T.Properties.VariableNames)
        error('Excel missing required column: %s', requiredCols{k});
    end
end

allRows = [];
sampleRows = [];

maxRows = min(height(T), cfg.maxRows);
for rowIdx = 1:maxRows
    baseParams = row_to_params(T(rowIdx, :));
    sampleId = sprintf('sample_%03d', rowIdx);
    sampleDir = fullfile(cfg.outputRoot, sampleId);
    ensure_folder(sampleDir);

    baselineDir = fullfile(sampleDir, 'baseline');
    ensure_folder(baselineDir);

    baseline = simulate_and_save(baseParams, baselineDir, 'baseline', cfg);
    baselineSummary = struct2table(flatten_metrics(sampleId, rowIdx, 'baseline', '', '', baseParams, baseParams, baseline.metrics, baseline.metrics));
    sampleRows = [sampleRows; baselineSummary]; %#ok<AGROW>

    perturbations = build_perturbations(baseParams, cfg);
    paramNames = fieldnames(perturbations);
    for p = 1:numel(paramNames)
        paramName = paramNames{p};
        pList = perturbations.(paramName);
        paramDir = fullfile(sampleDir, paramName);
        ensure_folder(paramDir);

        for j = 1:numel(pList)
            item = pList(j);
            outDir = fullfile(paramDir, item.folderName);
            ensure_folder(outDir);

            perturbed = simulate_and_save(item.params, outDir, item.label, cfg);
            compare = compare_patterns(baseline.gray, perturbed.gray, baseline.mask, perturbed.mask, baseline.metrics, perturbed.metrics);
            save_contour_overlay(baseline.mask, perturbed.mask, fullfile(outDir, 'binary_contour_compare.png'), ...
                sprintf('%s | %s', sampleId, item.label), cfg);
            save_paper_panel_assets(sampleId, item.label, baseline, perturbed, compare, sampleDir, outDir, cfg);

            rowStruct = flatten_metrics(sampleId, rowIdx, 'perturbation', paramName, item.label, baseParams, item.params, baseline.metrics, perturbed.metrics);
            rowStruct.pcc_vs_baseline = compare.pearson;
            rowStruct.ssim_vs_baseline = compare.ssim;
            rowStruct.delta_area_fraction = perturbed.metrics.area_fraction - baseline.metrics.area_fraction;
            rowStruct.delta_open_rate = perturbed.metrics.open_rate - baseline.metrics.open_rate;
            rowStruct.delta_components = perturbed.metrics.connected_components - baseline.metrics.connected_components;
            rowStruct.delta_holes = perturbed.metrics.hole_count - baseline.metrics.hole_count;
            rowStruct.delta_euler = perturbed.metrics.euler_number - baseline.metrics.euler_number;
            rowStruct.baseline_chirality_sign = baseline.metrics.chirality_sign;
            rowStruct.perturbed_chirality_sign = perturbed.metrics.chirality_sign;
            rowStruct.chirality_preserved = baseline.metrics.chirality_sign == perturbed.metrics.chirality_sign;
            rowStruct.delta_asymmetry = perturbed.metrics.asymmetry_index - baseline.metrics.asymmetry_index;

            rowTable = struct2table(rowStruct);
            allRows = [allRows; rowTable]; %#ok<AGROW>
        end
    end
end

summaryPath = fullfile(cfg.outputRoot, 'sensitivity_summary.xlsx');
if ~isempty(sampleRows)
    writetable(sampleRows, summaryPath, 'Sheet', 'baseline_samples');
end
if ~isempty(allRows)
    writetable(allRows, summaryPath, 'Sheet', 'perturbations');
    writetable(build_similarity_sheet(allRows), summaryPath, 'Sheet', 'similarity_only');
    writetable(build_similarity_wide_sheet(allRows), summaryPath, 'Sheet', 'similarity_wide');
end
writetable(config_table(cfg), summaryPath, 'Sheet', 'config');

disp(['Finished. Output saved to: ' cfg.outputRoot]);
end

function cfg = default_config()
cfg = struct();
cfg.excelPath = 'e:\A-Project-Codes\attention-cnn\20260304-new\sen-result\test-8-10.xlsx';
cfg.outputRoot = 'e:\A-Project-Codes\attention-cnn\20260304-new\sen-result\sensitivity_test_8_10';
cfg.matlabSourceDir = 'e:\A-Project-Codes\matlab-program';
cfg.BeadRes = 1;
cfg.SubstrateRes = 1;
cfg.DepRate = 30;
cfg.CenterDist = 500;
cfg.angleDeltasDeg = [-10, -5, -2, 2, 5, 10];
cfg.radScaleDeltas = [-0.10, -0.05, -0.02, 0.02, 0.05, 0.10];
cfg.timesCandidates = [1, 2, 3];
cfg.analyzeTimes = true;
cfg.figurePosition = [50, 50, 256, 444];
cfg.figureBgColor = [0.82, 0.88, 0.94];
cfg.chiralityTol = 1e-4;
cfg.maxRows = inf;
cfg.fontName = 'Times New Roman';
cfg.titleFontSize = 10;
cfg.legendFontSize = 9;
cfg.annotationFontSize = 9;
cfg.contourLineWidth = 1.3;
cfg.paperPanelWidth = 6.8;
cfg.paperPanelHeight = 5.6;
cfg.titleFontWeight = 'bold';
cfg.cleanOutputRoot = true;
cfg.saveLegacyPatternFigure = false;
cfg.saveBinaryOutlineFigure = false;
end

function P = row_to_params(rowTable)
P = struct();
P.times = double(rowTable.times(1));
P.ia = double(rowTable.ia(1));
P.phi1 = double(rowTable.phi1(1));
P.phi2 = double(rowTable.phi2(1));
P.phi3 = double(rowTable.phi3(1));
P.rad = double(rowTable.rad(1));
end

function perturbations = build_perturbations(baseParams, cfg)
perturbations = struct();
angleFields = {'ia','phi1','phi2','phi3'};

for i = 1:numel(angleFields)
    name = angleFields{i};
    items = repmat(struct('params', [], 'label', '', 'folderName', ''), 1, numel(cfg.angleDeltasDeg));
    for j = 1:numel(cfg.angleDeltasDeg)
        delta = cfg.angleDeltasDeg(j);
        P = baseParams;
        P.(name) = baseParams.(name) + delta;
        items(j).params = P;
        items(j).label = sprintf('%s_%+ddeg', name, delta);
        items(j).folderName = sprintf('%s_%sdeg', name, sanitize_delta(delta));
    end
    perturbations.(name) = items;
end

radItems = repmat(struct('params', [], 'label', '', 'folderName', ''), 1, numel(cfg.radScaleDeltas));
for j = 1:numel(cfg.radScaleDeltas)
    scaleDelta = cfg.radScaleDeltas(j);
    P = baseParams;
    P.rad = max(1, baseParams.rad * (1 + scaleDelta));
    pct = round(scaleDelta * 100);
    radItems(j).params = P;
    radItems(j).label = sprintf('rad_%+d%%', pct);
    radItems(j).folderName = sprintf('rad_%spct', sanitize_delta(pct));
end
perturbations.rad = radItems;

if cfg.analyzeTimes
    timeVals = setdiff(cfg.timesCandidates, clamp_times(baseParams.times));
    timeItems = repmat(struct('params', [], 'label', '', 'folderName', ''), 1, numel(timeVals));
    for j = 1:numel(timeVals)
        P = baseParams;
        P.times = timeVals(j);
        timeItems(j).params = P;
        timeItems(j).label = sprintf('times_%d', timeVals(j));
        timeItems(j).folderName = sprintf('times_%d', timeVals(j));
    end
    perturbations.times = timeItems;
end
end

function result = simulate_and_save(P, outDir, nameStem, cfg)
normalized = normalize_params(P);
[grayImg, mask] = generate_pattern(normalized, cfg);
metrics = analyze_pattern(grayImg, mask, cfg);

fig = figure('Visible', 'off', 'Color', cfg.figureBgColor, 'InvertHardCopy', 'off');
PlotSubstrate(cfg.CenterDist);
axis off;
axis normal;
colormap(gray(256));
set(fig, 'Position', cfg.figurePosition);
style_axes(gca, cfg);
if cfg.saveLegacyPatternFigure
    exportgraphics(fig, fullfile(outDir, [nameStem '_pattern.png']), 'Resolution', 300);
end
close(fig);

imwrite(grayImg, fullfile(outDir, [nameStem '_gray.png']));
imwrite(mask, fullfile(outDir, [nameStem '_binary.png']));
if cfg.saveBinaryOutlineFigure
    save_binary_outline(mask, fullfile(outDir, [nameStem '_binary_outline.png']), nameStem, cfg);
end

result = struct();
result.params = normalized;
result.gray = grayImg;
result.mask = mask;
result.metrics = metrics;
result.grayPath = fullfile(outDir, [nameStem '_gray.png']);
end

function [grayImg, mask] = generate_pattern(P, cfg)
Initialization(cfg.BeadRes, cfg.SubstrateRes, P.rad, cfg.CenterDist);
if P.times == 1
    OAD_Calculation(cfg.BeadRes, cfg.SubstrateRes, P.rad, cfg.CenterDist, cfg.DepRate, P.ia, P.phi1);
elseif P.times == 2
    OAD_Calculation(cfg.BeadRes, cfg.SubstrateRes, P.rad, cfg.CenterDist, cfg.DepRate, P.ia, P.phi1);
    OAD_Calculation(cfg.BeadRes, cfg.SubstrateRes, P.rad, cfg.CenterDist, cfg.DepRate, P.ia, P.phi2);
else
    OAD_Calculation(cfg.BeadRes, cfg.SubstrateRes, P.rad, cfg.CenterDist, cfg.DepRate, P.ia, P.phi1);
    OAD_Calculation(cfg.BeadRes, cfg.SubstrateRes, P.rad, cfg.CenterDist, cfg.DepRate, P.ia, P.phi2);
    OAD_Calculation(cfg.BeadRes, cfg.SubstrateRes, P.rad, cfg.CenterDist, cfg.DepRate, P.ia, P.phi3);
end

global h_sub;
H = double(h_sub);
grayImg = normalize_to_unit(H);
mask = H > max(H(:)) * 1e-9;
end

function metrics = analyze_pattern(grayImg, mask, cfg)
mask = logical(mask);
filled = imfill(mask, 'holes');
holesMask = filled & ~mask;
cc = bwconncomp(mask);
holeCC = bwconncomp(holesMask);

[nRows, nCols] = size(grayImg);
x = linspace(-1, 1, nCols);
X = repmat(x, nRows, 1);
mirrorResidual = grayImg - fliplr(grayImg);
signedMirror = sum(mirrorResidual .* X, 'all') / (sum(abs(grayImg(:))) + eps);

metrics = struct();
metrics.area_fraction = nnz(mask) / numel(mask);
metrics.open_rate = 1 - metrics.area_fraction;
metrics.connected_components = cc.NumObjects;
metrics.hole_count = holeCC.NumObjects;
metrics.euler_number = bweuler(mask);
metrics.chirality_score = signedMirror;
metrics.chirality_sign = sign_with_tol(signedMirror, cfg.chiralityTol);
metrics.asymmetry_index = sum(abs(mirrorResidual(:))) / (sum(abs(grayImg(:))) + eps);
end

function compare = compare_patterns(baseGray, pertGray, ~, ~, ~, ~)
compare = struct();
compare.pearson = safe_corr(baseGray, pertGray);
compare.ssim = safe_ssim(baseGray, pertGray);
end

function val = safe_corr(A, B)
a = double(A(:));
b = double(B(:));
if std(a) < eps || std(b) < eps
    val = NaN;
else
    C = corrcoef(a, b);
    val = C(1, 2);
end
end

function val = safe_ssim(A, B)
if exist('ssim', 'file') == 2
    val = ssim(A, B);
else
    val = NaN;
end
end

function P = normalize_params(P)
P.times = clamp_times(P.times);
P.ia = normalize_angle(P.ia);
P.phi1 = normalize_angle(P.phi1);
P.phi2 = normalize_angle(P.phi2);
P.phi3 = normalize_angle(P.phi3);
P.rad = max(1, double(P.rad));
end

function x = normalize_angle(x)
x = double(x);
if isnan(x) || isinf(x)
    x = 0;
end
x = mod(x, 360);
if abs(x - 360) < 1e-9
    x = 0;
end
if x < 0
    x = x + 360;
end
end

function t = clamp_times(t)
t = double(t);
if isnan(t) || isinf(t)
    t = 1;
end
t = round(t);
t = min(3, max(1, t));
end

function out = sanitize_delta(x)
if x >= 0
    out = ['p' num2str(abs(x))];
else
    out = ['m' num2str(abs(x))];
end
end

function save_binary_outline(mask, outPath, figTitle, cfg)
fig = figure('Visible', 'off', 'Color', 'w');
imshow(mask, []);
hold on;
B = bwboundaries(mask, 'noholes');
for k = 1:numel(B)
    plot(B{k}(:,2), B{k}(:,1), 'r', 'LineWidth', cfg.contourLineWidth);
end
t = title(strrep(figTitle, '_', '\_'));
style_text_object(t, cfg, cfg.titleFontSize);
style_axes(gca, cfg);
exportgraphics(fig, outPath, 'Resolution', 200);
close(fig);
end

function save_contour_overlay(baseMask, pertMask, outPath, ~, cfg)
fig = figure('Visible', 'off', 'Color', 'w');
imshow(zeros(size(baseMask)), []);
hold on;
baseB = bwboundaries(baseMask, 'noholes');
pertB = bwboundaries(pertMask, 'noholes');
for k = 1:numel(baseB)
    plot(baseB{k}(:,2), baseB{k}(:,1), 'r', 'LineWidth', cfg.contourLineWidth);
end
for k = 1:numel(pertB)
    plot(pertB{k}(:,2), pertB{k}(:,1), 'g', 'LineWidth', cfg.contourLineWidth);
end
style_axes(gca, cfg);
exportgraphics(fig, outPath, 'Resolution', 300);
close(fig);
end

function save_paper_panel_assets(~, label, baseline, perturbed, compare, sampleDir, outDir, cfg)
paperRoot = fullfile(sampleDir, 'paper_panels', folder_from_label(label));
ensure_folder(paperRoot);

baselineGrayBig = tile_to_full_pattern(baseline.gray);
perturbedGrayBig = tile_to_full_pattern(perturbed.gray);
baselineMaskBig = tile_to_full_pattern(baseline.mask);
perturbedMaskBig = tile_to_full_pattern(perturbed.mask);

save_full_image_panel(baselineGrayBig, fullfile(paperRoot, 'a_original_gray_full.png'), 'Original', cfg, false);
save_full_image_panel(perturbedGrayBig, fullfile(paperRoot, 'b_perturbed_gray_full.png'), pretty_label(label), cfg, false);
save_full_image_panel(baselineMaskBig, fullfile(paperRoot, 'c_original_binary_full.png'), 'Original binary', cfg, true);
save_full_image_panel(perturbedMaskBig, fullfile(paperRoot, 'd_perturbed_binary_full.png'), 'Perturbed binary', cfg, true);
save_full_overlay_panel(baselineMaskBig, perturbedMaskBig, compare, fullfile(paperRoot, 'e_contour_overlay_full.png'), cfg);
end

function row = flatten_metrics(sampleId, rowIdx, rowType, parameter, label, baseParams, pertParams, baseMetrics, pertMetrics)
row = struct();
row.sample_id = string(sampleId);
row.row_index = rowIdx;
row.row_type = string(rowType);
row.parameter = string(parameter);
row.label = string(label);
row.base_times = baseParams.times;
row.base_ia = baseParams.ia;
row.base_phi1 = baseParams.phi1;
row.base_phi2 = baseParams.phi2;
row.base_phi3 = baseParams.phi3;
row.base_rad = baseParams.rad;
row.pert_times = pertParams.times;
row.pert_ia = pertParams.ia;
row.pert_phi1 = pertParams.phi1;
row.pert_phi2 = pertParams.phi2;
row.pert_phi3 = pertParams.phi3;
row.pert_rad = pertParams.rad;
row.base_area_fraction = baseMetrics.area_fraction;
row.base_open_rate = baseMetrics.open_rate;
row.base_connected_components = baseMetrics.connected_components;
row.base_hole_count = baseMetrics.hole_count;
row.base_euler_number = baseMetrics.euler_number;
row.base_chirality_score = baseMetrics.chirality_score;
row.base_asymmetry_index = baseMetrics.asymmetry_index;
row.pert_area_fraction = pertMetrics.area_fraction;
row.pert_open_rate = pertMetrics.open_rate;
row.pert_connected_components = pertMetrics.connected_components;
row.pert_hole_count = pertMetrics.hole_count;
row.pert_euler_number = pertMetrics.euler_number;
row.pert_chirality_score = pertMetrics.chirality_score;
row.pert_asymmetry_index = pertMetrics.asymmetry_index;
end

function tbl = config_table(cfg)
names = {'excelPath';'outputRoot';'matlabSourceDir';'DepRate';'CenterDist';'angleDeltasDeg';'radScaleDeltas';'timesCandidates';'analyzeTimes';'chiralityTol'};
values = {cfg.excelPath; cfg.outputRoot; cfg.matlabSourceDir; num2str(cfg.DepRate); num2str(cfg.CenterDist); mat2str(cfg.angleDeltasDeg); mat2str(cfg.radScaleDeltas); mat2str(cfg.timesCandidates); mat2str(cfg.analyzeTimes); num2str(cfg.chiralityTol)};
tbl = table(names, values, 'VariableNames', {'name','value'});
end

function tbl = build_similarity_sheet(allRows)
varNames = allRows.Properties.VariableNames;
preferred = {'sample_id','row_index','parameter','label', ...
    'base_times','base_ia','base_phi1','base_phi2','base_phi3','base_rad', ...
    'pert_times','pert_ia','pert_phi1','pert_phi2','pert_phi3','pert_rad', ...
    'pcc_vs_baseline','ssim_vs_baseline', ...
    'delta_area_fraction','delta_open_rate','delta_components','delta_holes','delta_euler', ...
    'baseline_chirality_sign','perturbed_chirality_sign','chirality_preserved','delta_asymmetry'};
keep = preferred(ismember(preferred, varNames));
tbl = allRows(:, keep);
end

function wideTbl = build_similarity_wide_sheet(allRows)
wideTbl = table();
if isempty(allRows)
    return;
end

sampleIds = unique(allRows.sample_id, 'stable');
rowTables = cell(numel(sampleIds), 1);
allVarNames = {};

for i = 1:numel(sampleIds)
    sid = sampleIds(i);
    sub = allRows(allRows.sample_id == sid, :);
    row = table();
    row.sample_id = sid;
    row.row_index = sub.row_index(1);
    row.base_times = sub.base_times(1);
    row.base_ia = sub.base_ia(1);
    row.base_phi1 = sub.base_phi1(1);
    row.base_phi2 = sub.base_phi2(1);
    row.base_phi3 = sub.base_phi3(1);
    row.base_rad = sub.base_rad(1);

    for j = 1:height(sub)
        suffix = matlab.lang.makeValidName(char(sub.label(j)));
        row.(['pcc_' suffix]) = sub.pcc_vs_baseline(j);
        row.(['ssim_' suffix]) = sub.ssim_vs_baseline(j);
    end
    rowTables{i} = row;
    allVarNames = union(allVarNames, row.Properties.VariableNames, 'stable');
end

for i = 1:numel(rowTables)
    row = rowTables{i};
    missing = setdiff(allVarNames, row.Properties.VariableNames, 'stable');
    for j = 1:numel(missing)
        row.(missing{j}) = missing_value_for_column(missing{j}, height(row));
    end
    row = row(:, allVarNames);
    if isempty(wideTbl)
        wideTbl = row;
    else
        wideTbl = [wideTbl; row]; %#ok<AGROW>
    end
end
end

function val = missing_value_for_column(varName, nRows)
if startsWith(varName, 'pcc_') || startsWith(varName, 'ssim_') || startsWith(varName, 'base_')
    val = nan(nRows, 1);
elseif strcmp(varName, 'row_index')
    val = nan(nRows, 1);
else
    val = strings(nRows, 1);
end
end

function out = normalize_to_unit(H)
H = double(H);
hMin = min(H(:));
hMax = max(H(:));
if hMax - hMin < eps
    out = zeros(size(H));
else
    out = (H - hMin) ./ (hMax - hMin);
end
end

function style_axes(ax, cfg)
set(ax, 'FontName', cfg.fontName, 'FontSize', cfg.annotationFontSize, ...
    'LineWidth', 0.8, 'Box', 'off', 'LooseInset', max(ax.TightInset, 0.01));
end

function style_text_object(h, cfg, fontSize)
set(h, 'FontName', cfg.fontName, 'FontSize', fontSize, 'FontWeight', cfg.titleFontWeight);
end

function out = pretty_label(label)
out = strrep(label, 'phi', '\phi');
out = strrep(out, '_', ' ');
out = strrep(out, 'ia ', 'IA ');
out = strrep(out, 'rad ', 'Rad ');
out = strrep(out, 'times ', 'Times ');
end

function plot_boundaries(mask, colorValue, lineWidth)
B = bwboundaries(mask, 'noholes');
for k = 1:numel(B)
    plot(B{k}(:,2), B{k}(:,1), 'Color', colorValue, 'LineWidth', lineWidth);
end
end

function bigI = tile_to_full_pattern(I)
bigI = repmat(I, [3, 3]);
end

function save_full_image_panel(I, outPath, figTitle, cfg, isBinary)
fig = figure('Visible', 'off', 'Color', 'w');
fig.Units = 'inches';
fig.Position = [0.4, 0.4, cfg.paperPanelWidth, cfg.paperPanelHeight];
ax = axes(fig);
imshow(I, [], 'Parent', ax);
axis(ax, 'image');
axis(ax, 'off');
style_axes(ax, cfg);
t = title(ax, figTitle);
style_text_object(t, cfg, cfg.titleFontSize);
if isBinary
    colormap(ax, gray(2));
else
    colormap(ax, gray(256));
end
exportgraphics(fig, outPath, 'Resolution', 300);
close(fig);
end

function save_full_overlay_panel(baseMask, pertMask, compare, outPath, cfg)
fig = figure('Visible', 'off', 'Color', 'w');
fig.Units = 'inches';
fig.Position = [0.4, 0.4, cfg.paperPanelWidth, cfg.paperPanelHeight];
ax = axes(fig);
imshow(zeros(size(baseMask)), [], 'Parent', ax);
hold(ax, 'on');
plot_boundaries(baseMask, 'r', cfg.contourLineWidth);
plot_boundaries(pertMask, [0 0.7 0], cfg.contourLineWidth);
axis(ax, 'image');
axis(ax, 'off');
style_axes(ax, cfg);
exportgraphics(fig, outPath, 'Resolution', 300);
close(fig);
end

function apply_graphics_defaults(cfg)
set(groot, 'defaultAxesFontName', cfg.fontName);
set(groot, 'defaultTextFontName', cfg.fontName);
set(groot, 'defaultAxesFontSize', cfg.annotationFontSize);
set(groot, 'defaultTextInterpreter', 'none');
set(groot, 'defaultAxesTickLabelInterpreter', 'none');
set(groot, 'defaultLegendInterpreter', 'none');
end

function s = sign_with_tol(x, tol)
if abs(x) <= tol
    s = 0;
elseif x > 0
    s = 1;
else
    s = -1;
end
end

function ensure_folder(folderPath)
if exist(folderPath, 'dir') ~= 7
    mkdir(folderPath);
end
end

function prepare_output_root(folderPath, cleanFirst)
if exist(folderPath, 'dir') == 7 && cleanFirst
    rmdir(folderPath, 's');
end
ensure_folder(folderPath);
end

function cfg = merge_config(cfg, userCfg)
fields = fieldnames(userCfg);
for i = 1:numel(fields)
    cfg.(fields{i}) = userCfg.(fields{i});
end
end

function folder = folder_from_label(label)
tokens = split(label, '_');
paramName = char(tokens(1));
if strcmp(paramName, 'times')
    folder = label;
elseif strcmp(paramName, 'rad')
    pctStr = erase(label, 'rad_');
    pctStr = erase(pctStr, '%');
    folder = ['rad_' sanitize_delta(str2double(pctStr)) 'pct'];
else
    degStr = erase(label, [paramName '_']);
    degStr = erase(degStr, 'deg');
    folder = [paramName '_' sanitize_delta(str2double(degStr)) 'deg'];
end
end
