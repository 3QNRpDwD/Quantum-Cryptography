use rand::Rng;
use nalgebra as na;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Span, Line},
    widgets::{Axis, Block, Borders, Chart, Dataset, List, ListItem, Paragraph},
    Frame, Terminal,
};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::{io, thread};
use std::time::{Duration, Instant};
use clap::Parser;
use sha2::{Digest, Sha256};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of qubits to simulate
    #[arg(short, long, default_value_t = 1000)]
    qubits: usize,

    /// Whether Eve (eavesdropper) is active
    #[arg(short, long, default_value_t = false)]
    active_eve: bool,

    /// Error rate for eavesdropping (0.0 to 1.0)
    #[arg(short, long, default_value_t = 0.15)]
    error_rate: f64,
}

// KalmanFilter struct definition
struct KalmanFilter {
    x: na::Vector2<f64>,  // State estimate
    p: na::Matrix2<f64>,  // Estimate uncertainty
    q: na::Matrix2<f64>,  // Process uncertainty
    r: f64,               // Measurement uncertainty
}

impl KalmanFilter {
    fn new() -> Self {
        KalmanFilter {
            x: na::Vector2::new(0.0, 0.0),
            p: na::Matrix2::new(1.0, 0.0, 0.0, 1.0),
            q: na::Matrix2::new(1e-4, 0.0, 0.0, 1e-4),
            r: 1e-2,
        }
    }

    fn predict(&mut self) -> f64 {
        self.x = self.x;
        self.p = self.p + self.q;
        self.x[0]
    }

    fn update(&mut self, measurement: f64) {
        let y = measurement - self.x[0];
        let s = self.p[(0, 0)] + self.r;
        let k = self.p.column(0) / s;
        self.x += k * y;
        self.p -= k * na::RowVector2::new(self.p[(0, 0)], self.p[(1, 0)]);
    }
}

fn kalman_error_correction(alice_key: &[u8], bob_key: &[u8], kf: &mut KalmanFilter) -> (Vec<u8>, Vec<u8>) {
    let mut corrected_alice = alice_key.to_vec();
    let mut corrected_bob = bob_key.to_vec();

    for window in (0..alice_key.len()).step_by(100) {
        let end = std::cmp::min(window + 100, alice_key.len());
        let alice_window = &alice_key[window..end];
        let bob_window = &bob_key[window..end];

        let errors = alice_window.iter().zip(bob_window).filter(|&(a, b)| a != b).count();
        let error_rate = errors as f64 / alice_window.len() as f64;

        let predicted_error_rate = kf.predict();
        kf.update(error_rate);

        // 예측된 오류율을 기반으로 비트 수정
        for i in window..end {
            if (alice_key[i] != bob_key[i]) && (rand::random::<f64>() < predicted_error_rate) {
                corrected_bob[i] = alice_key[i]; // Bob의 키를 Alice의 키에 맞춰 수정
            }
        }
    }

    (corrected_alice, corrected_bob)
}

fn error_correction(alice_key: &[u8], bob_key: &[u8]) -> (Vec<u8>, Vec<u8>) {
    alice_key.iter().zip(bob_key).filter(|&(a, b)| a == b).map(|(&a, &b)| (a, b)).unzip()
}

fn privacy_amplification(key: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(key);
    hasher.finalize().to_vec()
}

fn security_parameter_estimation(error_rate: f64, threshold: f64) -> bool {
    error_rate < threshold
}

fn generate_random_bits(n: usize) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen_range(0..2) as u8).collect()
}

fn choose_random_bases(n: usize) -> Vec<char> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| if rng.gen() { '+' } else { 'x' }).collect()
}

fn measure_qubits(qubits: &[u8], bases: &[char], sender_bases: &[char]) -> Vec<u8> {
    qubits.iter().zip(bases.iter().zip(sender_bases)).map(|(&qubit, (&basis, &sender_basis))| {
        if basis == sender_basis {
            qubit
        } else {
            rand::thread_rng().gen_range(0..2) as u8
        }
    }).collect()
}

fn compare_bases(alice_bases: &[char], bob_bases: &[char]) -> Vec<usize> {
    alice_bases.iter().zip(bob_bases).enumerate()
        .filter_map(|(i, (&a, &b))| if a == b { Some(i) } else { None })
        .collect()
}

fn create_key(bits: &[u8], matched_indices: &[usize]) -> Vec<u8> {
    matched_indices.iter().map(|&i| bits[i]).collect()
}

fn simulate_eavesdropper(qubits: &[u8], error_rate: f64) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    qubits.iter().map(|&qubit| {
        if rng.gen::<f64>() > error_rate {
            qubit
        } else {
            1 - qubit
        }
    }).collect()
}

struct App {
    predictions: Vec<(f64, f64)>,
    actual_errors: Vec<(f64, f64)>,
    qubits_sent: usize,
    matching_bases: usize,
    alice_key: Vec<u8>,
    bob_key: Vec<u8>,
    keys_match: bool,
    eve_active: bool,
    error_rate: f64,
    final_error_rate: f64,
    prediction_error_rate: f64,
    prediction_success_rate: f64,
    kalman_log: Vec<String>,
    protocol_aborted: bool
}

fn main() -> Result<(), io::Error> {
    // Terminal initialization
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let app = run_simulation();
    let mut last_update = Instant::now();

    loop {
        terminal.draw(|f| ui(f, &app))?;

        if let Event::Key(key) = event::read()? {
            if let KeyCode::Char('q') = key.code {
                break;
            }
        }

        thread::sleep(Duration::from_millis(100));
    }

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    Ok(())
}

fn run_simulation() -> App {
    let args = Args::parse();

    let mut kf = KalmanFilter::new();
    let mut predictions = Vec::new();
    let mut actual_errors = Vec::new();
    let mut total_correct_predictions = 0;

    let alice_bits = generate_random_bits(args.qubits);
    let alice_bases = choose_random_bases(args.qubits);
    let transmitted_bits = match args.active_eve {
        true => simulate_eavesdropper(&alice_bits, args.error_rate),
        false => alice_bits.clone()
    };

    let bob_bases = choose_random_bases(args.qubits);
    let bob_measurements = measure_qubits(&transmitted_bits, &bob_bases, &alice_bases);
    let matched_indices = compare_bases(&alice_bases, &bob_bases);
    let alice_key = create_key(&alice_bits, &matched_indices);
    let bob_key = create_key(&bob_measurements, &matched_indices);

    for (i, (alice_window, bob_window)) in alice_key.windows(100).zip(bob_key.windows(100)).enumerate() {
        let errors = alice_window.iter().zip(bob_window).filter(|&(a, b)| a != b).count();
        let error_rate = errors as f64 / alice_window.len() as f64;

        let prediction = kf.predict();
        predictions.push((i as f64, prediction));
        actual_errors.push((i as f64, error_rate));

        if (prediction - error_rate).abs() < 0.01 {
            total_correct_predictions += 1;
        }

        kf.update(error_rate);
    }

    let prediction_error_rate = (actual_errors.iter().map(|&(_, e)| e).sum::<f64>() / actual_errors.len() as f64) * 100f64;
    let prediction_success_rate = total_correct_predictions as f64 / predictions.len() as f64;
    let errors = alice_key.iter().zip(&bob_key).filter(|&(a, b)| a != b).count();
    let error_rate = errors as f64 / alice_key.len() as f64;

    let mut kalman_log = Vec::new();
    for (i, (prediction, actual)) in predictions.iter().zip(actual_errors.iter()).enumerate() {
        kalman_log.push(format!("Step {}: P: {:.3}, A: {:.3}", i, prediction.1, actual.1));
    }

    // let (alice_corrected, bob_corrected) = error_correction(&alice_key, &bob_key);
    let (alice_corrected, bob_corrected) = kalman_error_correction(&alice_key, &bob_key, &mut kf);
    let final_alice_key = privacy_amplification(&alice_corrected);
    let final_bob_key = privacy_amplification(&bob_corrected);
    let final_errors = alice_corrected.iter().zip(&bob_corrected).filter(|&(a, b)| a != b).count();
    let final_error_rate = (final_errors as f64 / final_alice_key.len() as f64);
    let is_secure = security_parameter_estimation(final_error_rate / 100f64, 0.15);

    App {
        predictions,
        actual_errors,
        qubits_sent: args.qubits,
        matching_bases: matched_indices.len(),
        alice_key: final_alice_key[..8].to_vec(),
        bob_key:  final_bob_key[..8].to_vec(),
        keys_match: final_alice_key == final_bob_key,
        eve_active: args.active_eve,
        error_rate: error_rate * 100.0,
        final_error_rate,
        prediction_error_rate,
        prediction_success_rate,
        kalman_log,
        protocol_aborted: !is_secure
    }
}

fn render_graph(f: &mut Frame, app: &App, area: Rect) {
    let predicted_data: Vec<(f64, f64)> = app.predictions.iter().map(|&(x, y)| (x, y)).collect();
    let actual_data: Vec<(f64, f64)> = app.actual_errors.iter().map(|&(x, y)| (x, y)).collect();

    let datasets = vec![
        Dataset::default()
            .name("Predicted")
            .marker(symbols::Marker::Dot)  // Changed the marker for better distinction
            .style(Style::default().fg(Color::Magenta))  // Different color for the predicted line
            .data(&predicted_data),
        Dataset::default()
            .name("Actual")
            .marker(symbols::Marker::Braille)  // Changed the marker
            .style(Style::default().fg(Color::Yellow))  // Different color for the actual line
            .data(&actual_data),
    ];

    let chart = Chart::new(datasets)
        .block(Block::default().title("Error Rate Prediction").borders(Borders::ALL))
        .x_axis(
            Axis::default()
                .title("Window")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, app.predictions.len() as f64])
                .labels(vec![
                    Span::raw("0"),
                    Span::raw(format!("{}", app.predictions.len())),
                ]),
        )
        .y_axis(
            Axis::default()
                .title("Error Rate")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, 1.0])
                .labels(vec![
                    Span::raw("0.0"),
                    Span::raw("0.5"),
                    Span::raw("1.0"),
                ]),
        );
    f.render_widget(chart, area);
}



fn render_kalman_log(f: &mut Frame, app: &App, area: Rect) {
    let log_items: Vec<ListItem> = app.kalman_log
        .iter()
        .map(|s| ListItem::new(Line::from(Span::raw(s))))
        .collect();

    let log_list = List::new(log_items)
        .block(Block::default().title("Kalman Filter Log").borders(Borders::ALL))
        .style(Style::default().fg(Color::White))
        .highlight_style(Style::default().add_modifier(Modifier::ITALIC))
        .highlight_symbol(">>");

    f.render_widget(log_list, area);
}

fn render_simulation_info(f: &mut Frame, app: &App, area: Rect) {
    let info = vec![
        ("Qubits", format!("{} | Bases: {}", app.qubits_sent, app.matching_bases)),
        ("Match", format!("{} | Eve  : {}", app.keys_match, app.eve_active)),
        ("Error", format!("{:.3}% | Pred : {:.3}% | Final : {:.3}%"  , app.error_rate, app.prediction_error_rate, app.final_error_rate)),
        ("Pred success", format!("{:.4} | Aborted: {}", app.prediction_success_rate, app.protocol_aborted)),
    ];
    let max_left_label_length = info.iter().map(|(label, _)| label.len()).max().unwrap_or(0);
    let max_right_label_length = info.iter()
        .map(|(_, value)| {
            if let Some(pos) = value.find('|') {
                pos
            } else {
                value.len()
            }
        })
        .max()
        .unwrap_or(0);
    let info_text: Vec<Line> = info.iter()
        .map(|(label, value)| {
            if let Some(pos) = value.find('|') {
                let (left_part, right_part) = value.split_at(pos);
                let padded_left_part  = format!("{:<width_left$}", left_part.trim(), width_left = max_right_label_length);
                let padded_right_part = format!("{}{}"           , padded_left_part, right_part);
                let padded_label      = format!("{:<width$}"     , label, width = max_left_label_length);
                Line::from(Span::raw(format!("{}: {}", padded_label, padded_right_part)))
            } else {
                let padded_label = format!("{:<width$}", label, width = max_left_label_length);
                Line::from(Span::raw(format!("{}: {}", padded_label, value)))
            }
        })
        .collect();
    let info_paragraph = Paragraph::new(info_text)
        .block(Block::default().title("Sim Info").borders(Borders::ALL))
        .style(Style::default().fg(Color::Green))
        .alignment(ratatui::layout::Alignment::Left);

    f.render_widget(info_paragraph, area);
}
fn render_keys(f: &mut Frame, app: &App, area: Rect) {
    let keys_info = vec![
        format!(" Alice's Keys: {:?}", app.alice_key),
        format!(" Bob's   Keys: {:?}", app.bob_key),
    ];

    let keys_text: Vec<Line> = keys_info.iter()
        .map(|s| Line::from(Span::raw(s)))
        .collect();

    let keys_paragraph = Paragraph::new(keys_text)
        .block(Block::default().title("Keys").borders(Borders::ALL))
        .style(Style::default().fg(Color::Cyan))
        .alignment(ratatui::layout::Alignment::Left);

    f.render_widget(keys_paragraph, area);
}

fn ui(f: &mut Frame, app: &App) {
    let size = f.area();

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Ratio(6, 9),  // Top section (graph + log)
            Constraint::Ratio(3, 9),  // Bottom section (keys + info)
        ].as_ref())
        .split(size);

    let top_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Ratio(9, 12),  // Graph
            Constraint::Ratio(3, 12),   // Kalman log
        ].as_ref())
        .split(chunks[0]);

    let bottom_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Ratio(2, 5),  // Keys
            Constraint::Ratio(3, 5),  // Simulation info
        ].as_ref())
        .split(chunks[1]);

    let bottom_horizontal_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Ratio(5, 5),  // Keys and Simulation info
            Constraint::Ratio(3, 5),  // Extended Kalman log
        ].as_ref())
        .split(chunks[1]);

    let extended_log_area = Rect {
        x: top_chunks[1].x,
        y: top_chunks[1].y,
        width: top_chunks[1].width,
        height: top_chunks[1].height + bottom_horizontal_chunks[1].height,
    };

    render_graph(f, app, top_chunks[0]);
    render_kalman_log(f, app, top_chunks[1]);
    render_keys(f, app, bottom_horizontal_chunks[0].intersection(bottom_chunks[0]));
    render_simulation_info(f, app, bottom_horizontal_chunks[0].intersection(bottom_chunks[1]));
    render_kalman_log(f, app, extended_log_area);
}