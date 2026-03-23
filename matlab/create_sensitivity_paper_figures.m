function create_sensitivity_paper_figures(userCfg)
clc; close all;

cfg = default_plot_config();
if nargin >= 1 && ~isempty(userCfg)
    cfg = merge_struct(cfg, userCfg);
end

prepare_output_root(cfg.outputDir, cfg.cleanOutputDir);
apply_plot_defaults(cfg);

T = readtable(cfg.summaryExcel, 'Sheet', 'similarity_only');
if isempty(T)
    error('No rows found in similarity_only sheet: %s', cfg.summaryExcel);
end

T.sample_id = string(T.sample_id);
T.parameter = string(T.parameter);
T.label = string(T.label);

summaryTbl = build_parameter_summary(T);
writetable(summaryTbl, fullfile(cfg.outputDir, 'parameter_summary.xlsx'), 'Sheet', 'summary');

make_combined_similarity_figure(T, cfg);

make_geometry_figure(T, cfg);
make_summary_bar_figure(summaryTbl, cfg);

disp(['Paper figures saved to: ' cfg.outputDir]);
end

function cfg = default_plot_config()
cfg = struct();
cfg.summaryExcel = 'e:\A-Project-Codes\attention-cnn\20260304-new\sen-result\sensitivity_test_8_10\sensitivity_summary.xlsx';
cfg.outputDir = 'e:\A-Project-Codes\attention-cnn\20260304-new\sen-result\sensitivity_test_8_10\paper_figures_combined';
cfg.cleanOutputDir = true;
cfg.fontName = 'Times New Roman';
cfg.fontSize = 12;
cfg.titleFontSize = 14;
cfg.supertitleFontSize = 16;
cfg.lineWidth = 1.6;
cfg.markerSize = 6;
cfg.exportDPI = 600;
cfg.figureWidth = 13.5;
cfg.figureHeight = 5.8;
cfg.figureHeightGeom = 7.2;
cfg.figureHeightBar = 5.4;
cfg.axisColor = [0.15 0.15 0.15];
cfg.thresholdColor = [0.45 0.45 0.45];
cfg.sampleBarColors = [236 123 107; 116 163 212] / 255;
cfg.xtickAngle = 30;
cfg.showSubplotTags = false;
cfg.parametersOrdered = ["ia","phi1","phi2","phi3","rad","times"];
cfg.parameterDisplay = containers.Map( ...
    {'ia','phi1','phi2','phi3','rad','times'}, ...
    {'\bf\theta','\bf\phi_1','\bf\phi_2','\bf\phi_3','\bf{\itr}','\bf{\itn}'});
cfg.sampleDisplay = containers.Map( ...
    {'sample_001','sample_002'}, ...
    {'Chiral sample','Regular sample'});
cfg.colors = containers.Map( ...
    {'ia','phi1','phi2','phi3','rad','times'}, ...
    {[248 185 79]/255,[236 123 107]/255,[73 186 200]/255,[167 211 152]/255,[116 163 212]/255,[150 150 150]/255});
end

function make_combined_similarity_figure(T, cfg)
sampleOrder = ["sample_001","sample_001","sample_002","sample_002"];
metricOrder = ["pcc_vs_baseline","ssim_vs_baseline","pcc_vs_baseline","ssim_vs_baseline"];
rowLabels = ["Chiral sample PCC","Chiral sample SSIM","Regular sample PCC","Regular sample SSIM"];
params = cfg.parametersOrdered;
dataRows = table();
bottomAxes = gobjects(1, numel(params));

fig = figure('Visible', 'off', 'Color', 'w');
fig.Units = 'inches';
fig.Position = [0.3, 0.3, cfg.figureWidth, 10.8];
t = tiledlayout(4, numel(params), 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:4
    sampleId = sampleOrder(i);
    metricVar = metricOrder(i);
    metricLabel = upper(extractBefore(metricVar, "_vs"));
    for j = 1:numel(params)
        param = params(j);
        ax = nexttile;
        sub = T(T.sample_id == sampleId & T.parameter == param, :);
        if isempty(sub)
            axis(ax, 'off');
            continue;
        end

        [subSorted, xVals, xLabels] = x_values_for_parameter(sub, param);
        yVals = subSorted.(metricVar);
        dataRows = [dataRows; build_curve_export_rows(sampleId, param, metricVar, metricLabel, xVals, xLabels, yVals)]; %#ok<AGROW>

        if param == "times"
            stem(ax, xVals, yVals, 'filled', 'LineWidth', cfg.lineWidth, ...
                'MarkerSize', cfg.markerSize, 'Color', cfg.colors(char(param)));
            xlim(ax, [min(xVals)-0.4, max(xVals)+0.4]);
            xticks(ax, xVals);
            xticklabels(ax, xLabels);
            xline(ax, sub.base_times(1), ':', 'LineWidth', 0.9, 'Color', cfg.thresholdColor);
        else
            plot(ax, xVals, yVals, '-o', 'LineWidth', cfg.lineWidth, ...
                'MarkerSize', cfg.markerSize, 'Color', cfg.colors(char(param)), ...
                'MarkerFaceColor', cfg.colors(char(param)));
            xlim(ax, [min(xVals)-0.8, max(xVals)+0.8]);
            xticks(ax, xVals);
            xticklabels(ax, xLabels);
            xline(ax, 0, ':', 'LineWidth', 0.9, 'Color', cfg.thresholdColor);
        end

        if metricVar == "pcc_vs_baseline"
            yline(ax, 0.90, '--', 'LineWidth', 0.9, 'Color', cfg.thresholdColor);
        else
            yline(ax, 0.80, '--', 'LineWidth', 0.9, 'Color', cfg.thresholdColor);
        end
        ylim(ax, [0, 1]);
        yticks(ax, 0:0.2:1);
        grid(ax, 'off');
        style_axis(ax, cfg);

        if i == 1
            title(ax, cfg.parameterDisplay(char(param)), 'Interpreter', 'tex', ...
                'FontName', cfg.fontName, 'FontWeight', 'bold', 'FontSize', cfg.titleFontSize);
        end
        if j == 1
            ylabel(ax, rowLabels(i), 'FontName', cfg.fontName, ...
                'FontWeight', 'bold', 'FontSize', cfg.fontSize);
        end
        if i == 4
            xl = xlabel(ax, x_axis_label(param), 'FontName', cfg.fontName, ...
                'FontWeight', 'bold', 'FontSize', cfg.fontSize);
            set(xl, 'Units', 'normalized');
            bottomAxes(j) = ax;
        end
    end
end

drawnow;
for j = 1:numel(bottomAxes)
    if isgraphics(bottomAxes(j))
        xl = get(bottomAxes(j), 'XLabel');
        pos = get(xl, 'Position');
        pos(2) = -0.16;
        set(xl, 'Units', 'normalized', 'Position', pos, 'VerticalAlignment', 'top');
    end
end

title(t, 'PCC and SSIM under local parameter perturbation', ...
    'FontName', cfg.fontName, 'FontWeight', 'bold', 'FontSize', cfg.supertitleFontSize);
pngPath = fullfile(cfg.outputDir, 'Figure_R1_combined_PCC_SSIM.png');
pdfPath = fullfile(cfg.outputDir, 'Figure_R1_combined_PCC_SSIM.pdf');
exportgraphics(fig, pngPath, 'Resolution', cfg.exportDPI);
export_tiff(fig, replace(pngPath, '.png', '.tif'), cfg.exportDPI);
safe_export_pdf(fig, pdfPath);
close(fig);
writetable(dataRows, replace(pngPath, '.png', '_data.xlsx'), 'Sheet', 'data');
end

function make_geometry_figure(T, cfg)
sampleIds = unique(T.sample_id, 'stable');
params = cfg.parametersOrdered;
metrics = {'delta_area_fraction','delta_components','delta_holes','chirality_preserved'};
metricTitles = {'\Delta area fraction','\Delta components','\Delta holes','Chirality preserved'};
dataRows = table();

fig = figure('Visible', 'off', 'Color', 'w');
fig.Units = 'inches';
fig.Position = [0.3, 0.3, cfg.figureWidth, cfg.figureHeightGeom];
t = tiledlayout(numel(sampleIds), numel(metrics), 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:numel(sampleIds)
    sampleId = sampleIds(i);
    for j = 1:numel(metrics)
        ax = nexttile;
        M = nan(1, numel(params));
        for k = 1:numel(params)
            sub = T(T.sample_id == sampleId & T.parameter == params(k), :);
            if isempty(sub), continue; end
            if strcmp(metrics{j}, 'chirality_preserved')
                M(k) = mean(sub.chirality_preserved);
            else
                M(k) = max(abs(sub.(metrics{j})));
            end
        end
        dataRows = [dataRows; build_bar_export_rows(sampleId, metrics{j}, metricTitles{j}, params, M)]; %#ok<AGROW>

        bh = bar(ax, M, 'FaceColor', 'flat', 'EdgeColor', 'none');
        for k = 1:numel(params)
            bh.CData(k,:) = cfg.colors(char(params(k)));
        end
        xticks(ax, 1:numel(params));
        xticklabels(ax, display_labels(params, cfg));
        xtickangle(ax, cfg.xtickAngle);
        grid(ax, 'off');
        style_axis(ax, cfg);

        if strcmp(metrics{j}, 'chirality_preserved')
            ylim(ax, [0, 1.05]);
        else
            yline(ax, 0, ':', 'LineWidth', 0.8, 'Color', [0.4 0.4 0.4]);
        end

        title(ax, metricTitles{j}, 'Interpreter', 'tex', ...
            'FontName', cfg.fontName, 'FontWeight', 'bold', 'FontSize', cfg.titleFontSize);
        if j == 1
            ylabel(ax, sample_name(sampleId, cfg), 'FontName', cfg.fontName, ...
                'FontWeight', 'bold', 'FontSize', cfg.fontSize);
        end
    end
end

title(t, 'Structural robustness descriptors', ...
    'FontName', cfg.fontName, 'FontWeight', 'bold', 'FontSize', cfg.supertitleFontSize);
pngPath = fullfile(cfg.outputDir, 'Figure_R3_geometry_summary.png');
exportgraphics(fig, pngPath, 'Resolution', cfg.exportDPI);
export_tiff(fig, replace(pngPath, '.png', '.tif'), cfg.exportDPI);
safe_export_pdf(fig, fullfile(cfg.outputDir, 'Figure_R3_geometry_summary.pdf'));
close(fig);
writetable(dataRows, replace(pngPath, '.png', '_data.xlsx'), 'Sheet', 'data');
end

function make_summary_bar_figure(summaryTbl, cfg)
sampleIds = unique(summaryTbl.sample_id, 'stable');
params = cfg.parametersOrdered;
metrics = {'mean_ssim','mean_pcc','max_abs_delta_area','chirality_preserve_rate'};
metricTitles = {'Mean SSIM','Mean PCC','Max |\Delta area|','Chirality preserve rate'};
dataRows = table();

fig = figure('Visible', 'off', 'Color', 'w');
fig.Units = 'inches';
fig.Position = [0.3, 0.3, cfg.figureWidth, cfg.figureHeightBar];
t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for m = 1:numel(metrics)
    ax = nexttile;
    Y = nan(numel(params), numel(sampleIds));
    for i = 1:numel(sampleIds)
        for j = 1:numel(params)
            idx = summaryTbl.sample_id == sampleIds(i) & summaryTbl.parameter == params(j);
            if any(idx)
                Y(j, i) = summaryTbl.(metrics{m})(idx);
            end
        end
    end
    dataRows = [dataRows; build_grouped_bar_export_rows(sampleIds, params, metrics{m}, metricTitles{m}, Y)]; %#ok<AGROW>

        bh = bar(ax, Y, 'grouped', 'LineStyle', 'none');
        for i = 1:numel(sampleIds)
            bh(i).FaceColor = cfg.sampleBarColors(i, :);
            bh(i).EdgeColor = 'none';
        end
    xticks(ax, 1:numel(params));
    xticklabels(ax, display_labels(params, cfg));
    xtickangle(ax, cfg.xtickAngle);
    grid(ax, 'off');
    style_axis(ax, cfg);
    title(ax, metricTitles{m}, 'Interpreter', 'tex', ...
        'FontName', cfg.fontName, 'FontWeight', 'bold', 'FontSize', cfg.titleFontSize);
    if metrics{m} == "chirality_preserve_rate"
        ylim(ax, [0, 1.05]);
    end
    if m == 1
        lgd = legend(ax, arrayfun(@(s) sample_name(s, cfg), sampleIds, 'UniformOutput', false), ...
            'Location', 'southoutside', 'Orientation', 'horizontal', 'Box', 'off');
        set(lgd, 'FontName', cfg.fontName, 'FontSize', cfg.fontSize, 'FontWeight', 'bold');
    end
end

title(t, 'Parameter-wise sensitivity summary', ...
    'FontName', cfg.fontName, 'FontWeight', 'bold', 'FontSize', cfg.supertitleFontSize);
pngPath = fullfile(cfg.outputDir, 'Figure_R4_parameter_summary.png');
exportgraphics(fig, pngPath, 'Resolution', cfg.exportDPI);
export_tiff(fig, replace(pngPath, '.png', '.tif'), cfg.exportDPI);
safe_export_pdf(fig, fullfile(cfg.outputDir, 'Figure_R4_parameter_summary.pdf'));
close(fig);
writetable(dataRows, replace(pngPath, '.png', '_data.xlsx'), 'Sheet', 'data');
end

function summaryTbl = build_parameter_summary(T)
[G, sampleId, parameter] = findgroups(T.sample_id, T.parameter);
meanPCC = splitapply(@mean, T.pcc_vs_baseline, G);
meanSSIM = splitapply(@mean, T.ssim_vs_baseline, G);
minPCC = splitapply(@min, T.pcc_vs_baseline, G);
minSSIM = splitapply(@min, T.ssim_vs_baseline, G);
maxAbsArea = splitapply(@(x) max(abs(x)), T.delta_area_fraction, G);
maxAbsComp = splitapply(@(x) max(abs(x)), T.delta_components, G);
maxAbsHole = splitapply(@(x) max(abs(x)), T.delta_holes, G);
chiralityRate = splitapply(@mean, T.chirality_preserved, G);

summaryTbl = table(sampleId, parameter, meanPCC, meanSSIM, minPCC, minSSIM, ...
    maxAbsArea, maxAbsComp, maxAbsHole, chiralityRate, ...
    'VariableNames', {'sample_id','parameter','mean_pcc','mean_ssim','min_pcc','min_ssim', ...
    'max_abs_delta_area','max_abs_delta_components','max_abs_delta_holes','chirality_preserve_rate'});
end

function [subSorted, xVals, xLabels] = x_values_for_parameter(sub, param)
if param == "times"
    xVals = sub.pert_times;
    [xVals, idx] = sort(xVals);
    subSorted = sub(idx, :);
    xLabels = compose('%d', xVals);
else
    xVals = extract_signed_magnitude(sub.label, param);
    [xVals, idx] = sort(xVals);
    subSorted = sub(idx, :);
    xLabels = compose_signed_labels(xVals, param);
end
end

function vals = extract_signed_magnitude(labels, param)
vals = zeros(numel(labels), 1);
for i = 1:numel(labels)
    s = char(labels(i));
    s = erase(s, [char(param) '_']);
    s = erase(s, 'deg');
    s = erase(s, '%');
    vals(i) = str2double(s);
end
end

function labels = compose_signed_labels(vals, param)
labels = strings(size(vals));
for i = 1:numel(vals)
    if param == "rad"
        labels(i) = sprintf('%+g%%', vals(i));
    else
        labels(i) = sprintf('%+g', vals(i));
    end
end
end

function out = x_axis_label(param)
if param == "times"
    out = 'Target value';
elseif param == "rad"
    out = 'Perturbation';
else
    out = 'Perturbation (deg)';
end
end

function threshold = cfg_threshold(metricVar)
if strcmp(metricVar, 'ssim_vs_baseline')
    threshold = 0.8;
else
    threshold = 0.9;
end
end

function style_axis(ax, cfg)
set(ax, 'FontName', cfg.fontName, 'FontSize', cfg.fontSize, ...
    'FontWeight', 'bold', 'LineWidth', 1.0, 'Box', 'on', 'Layer', 'top', ...
    'XColor', cfg.axisColor, 'YColor', cfg.axisColor);
set(ax, 'XTickLabelRotation', cfg.xtickAngle);
xt = get(ax, 'XTickLabel');
set(ax, 'XTickLabel', xt);
end

function labels = display_labels(params, cfg)
labels = cell(size(params));
for i = 1:numel(params)
    labels{i} = cfg.parameterDisplay(char(params(i)));
end
end

function out = sample_name(sampleId, cfg)
key = char(sampleId);
if isKey(cfg.sampleDisplay, key)
    out = cfg.sampleDisplay(key);
else
    out = key;
end
end

function prepare_output_root(folderPath, cleanFirst)
if exist(folderPath, 'dir') == 7 && cleanFirst
    rmdir(folderPath, 's');
end
if exist(folderPath, 'dir') ~= 7
    mkdir(folderPath);
end
end

function apply_plot_defaults(cfg)
set(groot, 'defaultAxesFontName', cfg.fontName);
set(groot, 'defaultTextFontName', cfg.fontName);
set(groot, 'defaultAxesFontSize', cfg.fontSize);
set(groot, 'defaultAxesFontWeight', 'bold');
set(groot, 'defaultTextFontWeight', 'bold');
set(groot, 'defaultTextInterpreter', 'tex');
set(groot, 'defaultAxesTickLabelInterpreter', 'tex');
set(groot, 'defaultLegendInterpreter', 'tex');
end

function cfg = merge_struct(cfg, userCfg)
fields = fieldnames(userCfg);
for i = 1:numel(fields)
    cfg.(fields{i}) = userCfg.(fields{i});
end
end

function safe_export_pdf(fig, outPdf)
try
    exportgraphics(fig, outPdf, 'ContentType', 'vector');
catch ME
    warning('PDF export skipped for %s: %s', outPdf, ME.message);
end
end

function export_tiff(fig, outTif, dpi)
try
    exportgraphics(fig, outTif, 'Resolution', dpi);
catch ME
    warning('TIFF export skipped for %s: %s', outTif, ME.message);
end
end

function rows = build_curve_export_rows(sampleId, param, metricVar, metricLabel, xVals, xLabels, yVals)
n = numel(xVals);
rows = table(repmat(string(sampleId), n, 1), repmat(string(param), n, 1), ...
    repmat(string(metricVar), n, 1), repmat(string(metricLabel), n, 1), ...
    xVals(:), string(xLabels(:)), yVals(:), ...
    'VariableNames', {'sample_id','parameter','metric_var','metric_label','x_value','x_label','y_value'});
end

function rows = build_bar_export_rows(sampleId, metricVar, metricLabel, params, values)
n = numel(params);
rows = table(repmat(string(sampleId), n, 1), repmat(string(metricVar), n, 1), ...
    repmat(string(metricLabel), n, 1), string(params(:)), values(:), ...
    'VariableNames', {'sample_id','metric_var','metric_label','parameter','value'});
end

function rows = build_grouped_bar_export_rows(sampleIds, params, metricVar, metricLabel, Y)
rows = table();
for i = 1:numel(sampleIds)
    n = numel(params);
    one = table(repmat(string(sampleIds(i)), n, 1), repmat(string(metricVar), n, 1), ...
        repmat(string(metricLabel), n, 1), string(params(:)), Y(:, i), ...
        'VariableNames', {'sample_id','metric_var','metric_label','parameter','value'});
    rows = [rows; one]; %#ok<AGROW>
end
end
